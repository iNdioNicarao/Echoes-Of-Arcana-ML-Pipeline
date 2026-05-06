import Foundation
import Hub
import Tokenizers

/// Centralized manager for MLX tokenizers. 
/// Handles downloading/loading from the app bundle or local cache.
public actor TokenizerManager {
    
    public enum TokenizerError: Error {
        case modelNotFound
        case loadFailed(String)
        case encodingFailed
    }
    
    private var tokenizers: [String: Tokenizer] = [:]
    private var addedTokensMaps: [String: [String: Int]] = [:]
    
    // ODR support
    private var externalModelURL: URL?
    
    public init() {}
    
    public func setModelURL(_ url: URL) {
        self.externalModelURL = url
    }
    
    /// Loads the tokenizer for a specific model subdirectory if it's not already in memory.
    public func loadTokenizerIfNeeded(modelSubdir: String = "OracleModel.bundle") async throws {
        if tokenizers[modelSubdir] != nil { return }
        
        print("🧠 [TokenizerManager] Loading tokenizer for: \(modelSubdir)")
        
        var foundURL: URL? = nil
        
        // 1. Check for external ODR URL
        if let external = externalModelURL {
            foundURL = external
            print("📥 [TokenizerManager] Using ODR provided model URL.")
        } else {
            // 2. Fallback to bundle
            let possibleSubdirs = [modelSubdir, "Models/\(modelSubdir)", ""]
            for subdir in possibleSubdirs {
                if let url = Bundle.main.url(forResource: "tokenizer_config", withExtension: "json", subdirectory: subdir) {
                    foundURL = url.deletingLastPathComponent()
                    break
                }
            }
        }
        
        guard let modelURL = foundURL else {
            throw TokenizerError.modelNotFound
        }
        
        do {
            let tokenizer = try await AutoTokenizer.from(modelFolder: modelURL)
            self.tokenizers[modelSubdir] = tokenizer
            
            var tokensMap: [String: Int] = [:]
            
            // 1. Try to parse tokenizer_config.json's added_tokens_decoder map
            let configURL = modelURL.appendingPathComponent("tokenizer_config.json")
            if let configData = try? Data(contentsOf: configURL),
               let config = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
               let addedTokensDecoder = config["added_tokens_decoder"] as? [String: Any] {
                for (idString, tokenInfo) in addedTokensDecoder {
                    if let tokenDict = tokenInfo as? [String: Any],
                       let content = tokenDict["content"] as? String,
                       let id = Int(idString) {
                        tokensMap[content] = id
                    }
                }
            }
            
            // 2. Also try tokenizer.json's added_tokens array (more common in Qwen2.5)
            let tokenizerURL = modelURL.appendingPathComponent("tokenizer.json")
            if let tokenizerData = try? Data(contentsOf: tokenizerURL),
               let json = try? JSONSerialization.jsonObject(with: tokenizerData) as? [String: Any],
               let addedTokens = json["added_tokens"] as? [[String: Any]] {
                for tokenEntry in addedTokens {
                    if let content = tokenEntry["content"] as? String,
                       let id = tokenEntry["id"] as? Int {
                        tokensMap[content] = id
                    }
                }
            }
            
            self.addedTokensMaps[modelSubdir] = tokensMap
            
            print("✅ [TokenizerManager] Tokenizer loaded successfully. Added tokens: \(tokensMap.count)")
        } catch {
            throw TokenizerError.loadFailed(error.localizedDescription)
        }
    }
    
    public func encode(text: String, modelSubdir: String = "OracleModel.bundle") async throws -> [Int] {
        try await loadTokenizerIfNeeded(modelSubdir: modelSubdir)
        guard let tokenizer = tokenizers[modelSubdir] else { throw TokenizerError.modelNotFound }
        let addedTokensMap = addedTokensMaps[modelSubdir] ?? [:]
        
        // Parse the formatted text for special control tokens using Regex to ensure they are handled as single IDs
        // Qwen uses <|im_start|>, <|im_end|>, <|endoftext|>
        let pattern = "(<\\|im_start\\|>|<\\|im_end\\|>|<\\|endoftext\\|>)"
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return tokenizer.encode(text: text)
        }
        
        var tokenIds: [Int] = []
        let nsString = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsString.length))
        
        var lastIndex = 0
        for match in matches {
            if match.range.location > lastIndex {
                let chunk = nsString.substring(with: NSRange(location: lastIndex, length: match.range.location - lastIndex))
                // For chunks between special tokens, we must NOT add additional BOS/EOS tokens
                tokenIds.append(contentsOf: tokenizer.encode(text: chunk, addSpecialTokens: false))
            }
            let tag = nsString.substring(with: match.range)
            if let specialTokenId = addedTokensMap[tag] {
                tokenIds.append(specialTokenId)
            } else {
                // If not in map, fallback to standard encoding (might split it, which is bad for control tokens)
                tokenIds.append(contentsOf: tokenizer.encode(text: tag))
            }
            lastIndex = match.range.location + match.range.length
        }
        if lastIndex < nsString.length {
            let chunk = nsString.substring(with: NSRange(location: lastIndex, length: nsString.length - lastIndex))
            tokenIds.append(contentsOf: tokenizer.encode(text: chunk, addSpecialTokens: false))
        }
        
        return tokenIds
        }    
    public func decode(tokens: [Int], modelSubdir: String = "OracleModel.bundle") async throws -> String {
        try await loadTokenizerIfNeeded(modelSubdir: modelSubdir)
        guard let tokenizer = tokenizers[modelSubdir] else { throw TokenizerError.modelNotFound }
        
        // standard decode
        let baseDecoded = tokenizer.decode(tokens: tokens)
        
        // For Qwen, AutoTokenizer often handles decoding well, 
        // but if we see empty output for common tokens, we could use the map here too.
        return baseDecoded
    }
}
