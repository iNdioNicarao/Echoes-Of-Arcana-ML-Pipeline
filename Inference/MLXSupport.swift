import Foundation
import MLX
import MLXNN

// MARK: - Neural Network Support Components

/// A linear layer that handles 4-bit quantized weights with scales and biases.
class QuantizedLinear: Module {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let bias: MLXArray?
    
    let groupSize: Int
    let bits: Int
    
    init(_ inputDims: Int, _ outputDims: Int, bias: Bool = true, groupSize: Int = 64, bits: Int = 4) {
        self.groupSize = groupSize
        self.bits = bits
        
        // Shapes for 4-bit packed uint32
        let packedInDims = inputDims / (32 / bits)
        self.weight = MLXArray.zeros([outputDims, packedInDims], dtype: .uint32)
        self.scales = MLXArray.ones([outputDims, inputDims / groupSize], dtype: .bfloat16)
        self.biases = MLXArray.zeros([outputDims, inputDims / groupSize], dtype: .bfloat16)
        
        if bias {
            self.bias = MLXArray.zeros([outputDims], dtype: .bfloat16)
        } else {
            self.bias = nil
        }
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Standard MLX QuantizedLinear: out = x @ weight.T
        // Our weights are [Out, InPacked], so transpose: true computes x @ weight.T
        let out = MLX.quantizedMM(x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits)
        if let b = bias {
            return out + b
        }
        return out
    }
}

/// An embedding layer that handles 4-bit quantized weights.
class QuantizedEmbedding: Module {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    
    let groupSize: Int
    let bits: Int
    
    init(embeddingCount: Int, dimensions: Int, groupSize: Int = 64, bits: Int = 4) {
        self.groupSize = groupSize
        self.bits = bits
        
        let packedDims = dimensions / (32 / bits)
        self.weight = MLXArray.zeros([embeddingCount, packedDims], dtype: .uint32)
        self.scales = MLXArray.ones([embeddingCount, dimensions / groupSize], dtype: .bfloat16)
        self.biases = MLXArray.zeros([embeddingCount, dimensions / groupSize], dtype: .bfloat16)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let flatX = x.reshaped([-1])
        let numIndices = flatX.shape[0]
        let vocabSize = weight.shape[0]
        let hiddenDim = weight.shape[1] * (32 / bits)
        
        if numIndices == 0 {
            return MLXArray.zeros(shape + [hiddenDim], dtype: .bfloat16)
        }
        
        // Safety bounds check for token indices
        let safeX = MLX.maximum(MLXArray(Int32(0)), MLX.minimum(flatX.asType(.int32), MLXArray(Int32(vocabSize - 1))))
        
        // Vectorized gather and dequantize
        let pW = MLX.take(weight, safeX, axis: 0)
        let pS = MLX.take(scales, safeX, axis: 0)
        let pB = MLX.take(biases, safeX, axis: 0)
        
        let identity = MLXArray.eye(hiddenDim, dtype: .bfloat16)
        let dequantized = MLX.quantizedMM(identity, pW, scales: pS, biases: pB, transpose: true, groupSize: groupSize, bits: bits)
        
        let out = dequantized.transposed()
        return out.reshaped(shape + [hiddenDim])
    }
}

/// Rotary Positional Embedding (RoPE) implementation for Transformer models.
public class RoPE: Module {
    private let internalRoPE: MLXNN.RoPE
    
    public init(dimensions: Int, traditional: Bool = false, base: Float = 10000.0) {
        self.internalRoPE = MLXNN.RoPE(dimensions: dimensions, traditional: traditional, base: base)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        return internalRoPE(x, offset: offset)
    }
}

class FeedForwardNetwork: Module {
    let gate_proj: QuantizedLinear
    let down_proj: QuantizedLinear
    let up_proj: QuantizedLinear
    
    init(hiddenSize: Int, intermediateSize: Int) {
        self.gate_proj = QuantizedLinear(hiddenSize, intermediateSize, bias: false, groupSize: 64, bits: 4)
        self.up_proj = QuantizedLinear(hiddenSize, intermediateSize, bias: false, groupSize: 64, bits: 4)
        self.down_proj = QuantizedLinear(intermediateSize, hiddenSize, bias: false, groupSize: 64, bits: 4)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = MLXNN.silu(gate_proj(x))
        let up = up_proj(x)
        return down_proj(gate * up)
    }
}
