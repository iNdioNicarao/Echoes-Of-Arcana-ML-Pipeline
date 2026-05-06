import Foundation
import MLX
import MLXNN

/// Manages the lifecycle and weight mapping for the local Qwen Text interpretation model.
public actor LocalInterpretationManager {
    
    public enum ModelState: Equatable {
        case unloaded
        case loading
        case ready
        case error(String)
    }
    
    public private(set) var state: ModelState = .unloaded
    private var model: QwenTextModel?
    private let modelSubdir = "OracleModel.bundle"
    
    // Structured Concurrency: Sharable loading task
    private var loadingTask: Task<QwenTextModel, Error>?
    
    // ODR support
    private var externalModelURL: URL?
    
    public init() {
        print("🧠 [LocalInterpretationManager] Initialized.")
    }
    
    public func setModelURL(_ url: URL) {
        self.externalModelURL = url
    }
    
    public func loadModelIfNeeded() async throws -> QwenTextModel {
        if case .ready = state, let m = model { return m }
        
        // If already loading, return the existing task
        if let existingTask = loadingTask {
            return try await existingTask.value
        }
        
        let newTask = Task {
            try await performLoad()
        }
        
        loadingTask = newTask
        
        do {
            let loadedModel = try await newTask.value
            loadingTask = nil // Clear after success
            return loadedModel
        } catch {
            loadingTask = nil // Clear on failure to allow retry
            throw error
        }
    }
    
    private func performLoad() async throws -> QwenTextModel {
        state = .loading
        LogService.shared.info("🧠 Starting Text Model load process...")
        LogService.shared.breadcrumb(name: "Model Load Start", category: "mlx", data: ["model": modelSubdir])
        
        do {
            var foundModelURL: URL? = nil
            
            // 1. Check if we have an external ODR URL injected
            if let external = externalModelURL {
                foundModelURL = external
                LogService.shared.info("📥 [MLX] Using ODR provided model URL.")
            } else {
                // 2. Fallback to bundle check (only for non-ODR debug builds)
                let possibleSubdirs = [modelSubdir, "Models/\(modelSubdir)", ""]
                for subdir in possibleSubdirs {
                    if let url = Bundle.main.url(forResource: "config", withExtension: "json", subdirectory: subdir) {
                        foundModelURL = url.deletingLastPathComponent()
                        break
                    }
                }
            }
            
            guard let modelURL = foundModelURL else {
                let err = NSError(domain: "MLX", code: 404, userInfo: [NSLocalizedDescriptionKey: "Text model files not found."])
                LogService.shared.error(err, message: "Model directory not found in bundle.")
                throw err
            }
            
            let configURL = modelURL.appendingPathComponent("config.json")
            let configData = try Data(contentsOf: configURL)
            let qwenConfig = try JSONDecoder().decode(QwenTextConfig.self, from: configData)
            
            let textModel = QwenTextModel(config: qwenConfig)
            let weightsURL = modelURL.appendingPathComponent("model_int4.safetensors")
            
            let rawArrays: [String: MLXArray]
            if !FileManager.default.fileExists(atPath: weightsURL.path) {
                let fallbackURL = modelURL.appendingPathComponent("model.safetensors")
                if FileManager.default.fileExists(atPath: fallbackURL.path) {
                    rawArrays = try MLX.loadArrays(url: fallbackURL)
                } else {
                    let err = NSError(domain: "MLX", code: 405, userInfo: [NSLocalizedDescriptionKey: "Weights file not found."])
                    LogService.shared.error(err, message: "Weight files (.safetensors) missing from bundle.")
                    throw err
                }
            } else {
                rawArrays = try MLX.loadArrays(url: weightsURL)
            }
            
            LogService.shared.breadcrumb(name: "Weights Loaded", category: "mlx", data: ["count": rawArrays.count])
            
            var mappedArrays: [String: MLXArray] = [:]
            for (key, array) in rawArrays {
                var newKey = key
                if key.hasPrefix("model.") {
                    newKey = String(key.dropFirst(6))
                }
                
                // MLX weight update is picky about dtypes.
                if key.contains(".weight") {
                    if array.dtype == .uint32 {
                        mappedArrays[newKey] = array
                    } else {
                        mappedArrays[newKey] = array.asType(.bfloat16)
                    }
                } else if key.contains(".bias") || key.contains(".scales") || key.contains(".biases") {
                    // Both .bias (standard) and .biases (quantization) are mapped
                    mappedArrays[newKey] = array.asType(.bfloat16)
                } else {
                    mappedArrays[newKey] = array.asType(.bfloat16)
                }
            }
            
            // Handle tied word embeddings
            if mappedArrays["lm_head.weight"] == nil && mappedArrays["embed_tokens.weight"] != nil {
                LogService.shared.debug("🔗 Tying lm_head weights to embed_tokens.")
                mappedArrays["lm_head.weight"] = mappedArrays["embed_tokens.weight"]
                mappedArrays["lm_head.scales"] = mappedArrays["embed_tokens.scales"]
                mappedArrays["lm_head.biases"] = mappedArrays["embed_tokens.biases"]
                
                if let embedBias = mappedArrays["embed_tokens.bias"] {
                    mappedArrays["lm_head.bias"] = embedBias
                }
            }
            
            // Explicitly evaluate all realizing arrays to ensure Metal memory is correctly populated before update
            MLX.eval(Array(mappedArrays.values))
            
            textModel.update(parameters: ModuleParameters.unflattened(mappedArrays))
            
            self.model = textModel
            self.state = .ready
            LogService.shared.info("✅ Text Model loaded successfully.")
            return textModel
        } catch {
            LogService.shared.error(error, message: "Model loading failed.")
            self.state = .error(error.localizedDescription)
            throw error
        }
    }
    
    public func unloadModel() {
        self.model = nil
        self.state = .unloaded
        MLX.Memory.clearCache()
        print("🧠 [LocalInterpretationManager] Text Model unloaded.")
    }
}
