import Foundation
import MLX
import MLXNN

// MARK: - Qwen Text Configuration
public struct QwenTextConfig: Codable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let vocabSize: Int
    public let intermediateSize: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
    }
}

// MARK: - Qwen Text Model
public class QwenTextModel: Module {
    public let config: QwenTextConfig
    
    let embed_tokens: QuantizedEmbedding
    let layers: [DecoderLayer]
    let norm: MLXNN.RMSNorm
    let lm_head: QuantizedLinear
    
    public init(config: QwenTextConfig) {
        self.config = config
        // Using QuantizedEmbedding since weights are 4-bit quantized
        self.embed_tokens = QuantizedEmbedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize, groupSize: 64, bits: 4)
        self.layers = (0..<config.numHiddenLayers).map { _ in
            DecoderLayer(config: config)
        }
        self.norm = MLXNN.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps) 
        self.lm_head = QuantizedLinear(config.hiddenSize, config.vocabSize, bias: false, groupSize: 64, bits: 4)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        var h = embed_tokens(x)
        
        // --- OPTIMIZATION: Create causal mask once if missing and sequence length > 1 ---
        var effectiveMask = mask
        if effectiveMask == nil && x.shape[1] > 1 {
            effectiveMask = MLXNN.MultiHeadAttention.createAdditiveCausalMask(x.shape[1], dtype: h.dtype)
        }

        for (i, layer) in layers.enumerated() {
            // CRITICAL: Stop submitting GPU work immediately if task is cancelled (e.g. backgrounding)
            if Task.isCancelled { break }
            
            h = layer(h, mask: effectiveMask, cache: cache, layerIdx: i)
            
            // --- MEMORY STABILITY: Evaluate each layer during heavy prompt processing ---
            // This prevents the graph from growing too large and spiking memory at the end.
            if x.shape[1] > 1 {
                MLX.eval(h)
            }
        }
        return lm_head(norm(h))
    }
    
    /// Incremental generation helper: returns logits for the next token
    public func generateNextTokenLogits(tokenID: Int, cache: KVCache) -> MLXArray {
        let x = MLXArray([Int32(tokenID)]).expandedDimensions(axis: 0)
        let logits = self(x, mask: nil, cache: cache)
        return logits[0, -1, 0...]
    }
}

// MARK: - Decoder Layer
class DecoderLayer: Module {
    let self_attn: TextAttention
    let mlp: FeedForwardNetwork
    let input_layernorm: MLXNN.RMSNorm
    let post_attention_layernorm: MLXNN.RMSNorm
    
    init(config: QwenTextConfig) {
        self.self_attn = TextAttention(dimensions: config.hiddenSize, numHeads: config.numAttentionHeads, numKVHeads: config.numKeyValueHeads)
        self.mlp = FeedForwardNetwork(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        self.input_layernorm = MLXNN.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.post_attention_layernorm = MLXNN.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil, layerIdx: Int? = nil) -> MLXArray {
        var r = input_layernorm(x)
        r = self_attn(r, mask: mask, cache: cache, layerIdx: layerIdx)
        let h = x + r
        
        r = post_attention_layernorm(h)
        r = mlp(r)
        return h + r
    }
}

// MARK: - Text-Specific Attention (Matches Qwen2.5 Text Weights)
class TextAttention: Module {
    let q_proj: QuantizedLinear
    let k_proj: QuantizedLinear
    let v_proj: QuantizedLinear
    let o_proj: QuantizedLinear
    let rope: RoPE
    
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    
    init(dimensions: Int, numHeads: Int, numKVHeads: Int) {
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = dimensions / numHeads
        
        self.q_proj = QuantizedLinear(dimensions, numHeads * headDim, bias: true, groupSize: 64, bits: 4)
        self.k_proj = QuantizedLinear(dimensions, numKVHeads * headDim, bias: true, groupSize: 64, bits: 4)
        self.v_proj = QuantizedLinear(dimensions, numKVHeads * headDim, bias: true, groupSize: 64, bits: 4)
        self.o_proj = QuantizedLinear(numHeads * headDim, dimensions, bias: false, groupSize: 64, bits: 4)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: 1_000_000)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil, layerIdx: Int? = nil) -> MLXArray {
        let B = x.shape[0]
        let L = x.shape[1]
        
        var q = q_proj(x).reshaped(B, L, numHeads, headDim)
        var k = k_proj(x).reshaped(B, L, numKVHeads, headDim)
        var v = v_proj(x).reshaped(B, L, numKVHeads, headDim)
        
        // --- FIX: Transpose FIRST, then apply RoPE ---
        // New shapes: [B, Heads, L, Dim]
        q = q.transposed(0, 2, 1, 3).contiguous()
        k = k.transposed(0, 2, 1, 3).contiguous()
        v = v.transposed(0, 2, 1, 3).contiguous()

        let offset: Int
        if let idx = layerIdx, idx < (cache?.keys.count ?? 0) {
            // Cache is stored as [B, Heads, L_total, Dim]
            offset = cache?.keys[idx]?.shape[2] ?? 0
        } else {
            offset = 0
        }
        
        // MLX RoPE applies to the second-to-last dimension (index 2 in [B, H, L, D]), which is correctly L.
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)
        
        if let cache = cache, let idx = layerIdx {
            // Ensure pre-allocation if cache was empty-init
            while cache.keys.count <= idx {
                cache.keys.append(nil)
                cache.values.append(nil)
            }
            
            if let prevK = cache.keys[idx], let prevV = cache.values[idx] {
                k = MLX.concatenated([prevK, k], axis: 2)
                v = MLX.concatenated([prevV, v], axis: 2)
            }
            cache.keys[idx] = k
            cache.values[idx] = v
        }
        
        // --- GQA SUPPORT ---
        if numKVHeads != numHeads {
            let nRepeats = numHeads / numKVHeads
            let L_total = k.shape[2]
            k = MLX.repeated(k.expandedDimensions(axis: 2), count: nRepeats, axis: 2)
                 .reshaped(B, numHeads, L_total, headDim)
            v = MLX.repeated(v.expandedDimensions(axis: 2), count: nRepeats, axis: 2)
                 .reshaped(B, numHeads, L_total, headDim)
        }
        
        let scale = 1.0 / sqrt(Float(headDim))

        // Scale early to prevent float16 overflow during matmul accumulation.
        // This keeps intermediate dot-products within the ~65k limit of float16.
        let q_scaled = q * scale
        var scores = MLX.matmul(q_scaled, k.transposed(0, 1, 3, 2))

        if let mask = mask {
            scores = scores + mask
        }

        // Softmax remains in float32 for numerical stability
        let weights = MLX.softmax(scores.asType(.float32), axis: -1).asType(q.dtype)
        let output = MLX.matmul(weights, v)

        
        let out = output.transposed(0, 2, 1, 3).reshaped(B, L, numHeads * headDim).contiguous()
        return o_proj(out)
    }
}

extension MLXArray {
    func isNanSafe() -> Bool {
        let s = MLX.sum(self).item(Float.self)
        return !s.isNaN && !s.isInfinite
    }
}

public class KVCache {
    public var keys: [MLXArray?]
    public var values: [MLXArray?]
    
    public init() {
        self.keys = []
        self.values = []
    }
    
    public init(numLayers: Int) {
        self.keys = Array(repeating: nil, count: numLayers)
        self.values = Array(repeating: nil, count: numLayers)
    }
}

