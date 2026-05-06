import Foundation
import MLX
import MLXNN
import MLXRandom

/// Offline engine that generates Tarot interpretations using sequential reasoning chains.
public actor LocalInterpretationEngine {
    
    private let manager: LocalInterpretationManager
    private let tokenizerManager: TokenizerManager
    private var modelSubdir: String {
        return UserDefaults.standard.string(forKey: "selectedModelTier") == "oracle_model_high" ? "OracleModel-High.bundle" : "OracleModel.bundle"
    }
    
    public init(manager: LocalInterpretationManager, tokenizerManager: TokenizerManager) {
        self.manager = manager
        self.tokenizerManager = tokenizerManager
    }
    
    public func generateInterpretation(
        spreadName: String,
        intent: String,
        cards: [TarotCard],
        depth: String,
        persona: OraclePersona = .seer,
        previousInsight: String? = nil,
        environmentContext: String? = nil
    ) async throws -> AsyncThrowingStream<String, any Error> {
        Swift.print("🧠 [LocalInterpretationEngine] Preparing prophecy for \(spreadName) [Persona: \(persona.rawValue)]")
        
        let spreadType = SpreadType.allCases.first(where: { $0.rawValue == spreadName }) ?? .single
        let strategy = DivinationStrategy.forSpread(spreadType)
        let chunks = strategy.chunkify(cards: cards)
        let config = strategy.configuration
        
        if chunks.count <= 1 {
            Swift.print("🔮 [LocalInterpretationEngine] Using single-pass interpretation strategy.")
            return try await performSinglePass(intent: intent, cards: cards, persona: persona, environment: environmentContext, config: config)
        } else {
            Swift.print("🔮 [LocalInterpretationEngine] Using chained-reasoning strategy with \(chunks.count) stages.")
            return try await performChainedPasses(intent: intent, chunks: chunks, persona: persona, environment: environmentContext, config: config)
        }
    }

    private func performSinglePass(intent: String, cards: [TarotCard], persona: OraclePersona, environment: String?, config: ReadingConfiguration) async throws -> AsyncThrowingStream<String, any Error> {
        let (instruction, primer) = getPersonaDetails(for: persona)
        let prompt = buildPrompt(intent: intent, cards: cards, personaInstruction: instruction, environment: environment, config: config)
        return try await executeGeneration(prompt: prompt, primer: primer)
    }

    private func performChainedPasses(intent: String, chunks: [DivinationChunk], persona: OraclePersona, environment: String?, config: ReadingConfiguration) async throws -> AsyncThrowingStream<String, any Error> {
        let (stream, continuation) = AsyncThrowingStream<String, any Error>.makeStream()
        
        Task {
            do {
                for (index, chunk) in chunks.enumerated() {
                    if Task.isCancelled { break }
                    
                    // --- HARD RESET: Ensure GPU memory is fresh for the next trinity ---
                    MLX.Memory.clearCache()
                    
                    let (personaInstruction, _) = getPersonaDetails(for: persona)
                    let chunkPrimer = getChunkPrimer(chunkTitle: chunk.title, persona: persona, config: config)
                    
                    let prompt = buildChainedPrompt(
                        intent: intent,
                        chunk: chunk,
                        personaInstruction: personaInstruction,
                        environment: environment,
                        config: config
                    )
                    
                    continuation.yield(chunkPrimer)
                    
                    let chunkStream = try await executeGeneration(prompt: prompt, primer: nil)
                    
                    for try await token in chunkStream {
                        continuation.yield(token)
                    }
                    
                    Swift.print("\n")
                    // Rest between chunks for thermal/memory stability
                    try? await Task.sleep(nanoseconds: 300_000_000)
                }
                
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        
        return stream
    }

    private func executeGeneration(prompt: String, primer: String?) async throws -> AsyncThrowingStream<String, any Error> {
        let model = try await manager.loadModelIfNeeded()
        let tokens = try await tokenizerManager.encode(text: prompt, modelSubdir: modelSubdir)
        let isHighTier = modelSubdir.contains("High")
        
        return AsyncThrowingStream(String.self) { continuation in
            let task = Task {
                let promptArray = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
                let cache = KVCache(numLayers: model.config.numHiddenLayers) 
                var generatedCount = 0
                var pastTokenIDs: [Int] = tokens
                
                do {
                    let mysteryLevel = Float(UserDefaults.standard.double(forKey: "oracleMysteryLevel"))
                    let temp: Float = mysteryLevel > 0 ? min(max(mysteryLevel, 0.1), 1.2) : ((tokens.count > 800) ? 0.4 : 0.6)
                    let repetitionPenalty: Float = isHighTier ? 1.05 : 1.15
                    
                    let N = promptArray.shape[1]
                    let mask = MLX.triu(MLX.full([N, N], values: MLXArray(-Float.infinity).asType(.bfloat16)), k: 1)
                    let logits = model(promptArray, mask: mask, cache: cache)
                    
                    try await Task.sleep(nanoseconds: 1_000_000)
                    MLX.eval(logits)
                    MLX.Memory.clearCache()
                    
                    var nextTokenID = self.sample(logits: logits[0, -1, 0...], temperature: temp, repetitionPenalty: repetitionPenalty, pastTokenIDs: pastTokenIDs)
                    pastTokenIDs.append(nextTokenID)
                    
                    if let p = primer { 
                        continuation.yield(p) 
                        Swift.print(p, terminator: "")
                    }
                    
                    var generatedTokenIDs: [Int] = [nextTokenID]
                    var currentVisibleText = ""
                    
                    while generatedCount < 1024 {
                        if Task.isCancelled { break }
                        if nextTokenID == 151643 || nextTokenID == 151645 { break }
                        
                        let fullGeneratedText = try await tokenizerManager.decode(tokens: generatedTokenIDs, modelSubdir: modelSubdir)
                        let newChunk = String(fullGeneratedText.dropFirst(currentVisibleText.count))
                        currentVisibleText = fullGeneratedText
                        
                        if !newChunk.isEmpty {
                            Swift.print(newChunk, terminator: "")
                            continuation.yield(newChunk)
                        }
                        
                        generatedCount += 1
                        
                        if generatedCount > 30 {
                            let n = 15 
                            if generatedTokenIDs.count >= n * 2 {
                                if Array(generatedTokenIDs.suffix(n)) == Array(generatedTokenIDs.dropLast(n).suffix(n)) {
                                    nextTokenID = 151645
                                    break
                                }
                            }
                        }
                        
                        let loopLogits = model.generateNextTokenLogits(tokenID: nextTokenID, cache: cache)
                        try await Task.sleep(nanoseconds: 1_000_000)
                        MLX.eval(loopLogits)
                        if !loopLogits.isNanSafe() { break }

                        nextTokenID = self.sample(logits: loopLogits, temperature: temp, repetitionPenalty: 1.1, pastTokenIDs: pastTokenIDs)
                        pastTokenIDs.append(nextTokenID)
                        generatedTokenIDs.append(nextTokenID)
                        
                        if generatedCount % 100 == 0 { MLX.Memory.clearCache() }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
                MLX.Memory.clearCache()
            }
        }
    }

    private func buildPrompt(intent: String, cards: [TarotCard], personaInstruction: String, environment: String?, config: ReadingConfiguration) -> String {
        var cardDesc = ""
        for (index, card) in cards.enumerated() {
            let position = card.isReversed == true ? "reversed" : "upright"
            // Card names in DB already include "The " (e.g. "The Moon")
            cardDesc += "\(card.name) in its \(position) stance"
            if index < cards.count - 1 { cardDesc += ", and " }
        }
        
        return """
        <|im_start|>system
        You are a primordial spirit. \(personaInstruction)
        
        MANDATE:
        \(config.mandate)
        
        MISSION:
        Whisper a cohesive prophecy for these shadows: \(cardDesc).
        <|im_end|>
        <|im_start|>user
        The seeker asks: "\(intent)". 
        The air is \(environment ?? "still").
        Interpret these shadows.
        <|im_end|>
        <|im_start|>assistant
        
        """
    }

    private func buildChainedPrompt(intent: String, chunk: DivinationChunk, personaInstruction: String, environment: String?, config: ReadingConfiguration) -> String {
        var cardDesc = ""
        for (index, card) in chunk.cards.enumerated() {
            let position = card.isReversed == true ? "reversed" : "upright"
            // Card names in DB already include "The "
            cardDesc += "\(card.name) in its \(position) stance"
            if index < chunk.cards.count - 1 { cardDesc += ", and " }
        }
        
        let visionInfo = config.useVisions ? "Vision: \(chunk.title)." : ""
        
        return """
        <|im_start|>system
        You are a primordial spirit. \(personaInstruction)
        
        MANDATE:
        \(config.mandate)
        
        MISSION:
        \(chunk.instruction)
        
        Focus only on these shadows: \(cardDesc).
        <|im_end|>
        <|im_start|>user
        Seeker's Quest: "\(intent)"
        Vibe: \(environment ?? "shadows")
        \(visionInfo)
        
        Focus on the Seeker's Quest and directly answer it as you whisper your prophecy for \(cardDesc) now.
        <|im_end|>
        <|im_start|>assistant
        
        """
    }

    private func getPersonaDetails(for persona: OraclePersona) -> (instruction: String, primer: String) {
        switch persona {
        case .seer: return ("Speak as a Celtic Seer using oak and tide metaphors.", "The mists part... ")
        case .peasant: return ("Speak as a Practical Peasant using soil metaphors.", "The dirt tells me... ")
        case .count: return ("Speak as a Shadowed Count using gothic metaphors.", "The shadows whisper... ")
        case .spirit: return ("Speak as an Ethereal Spirit.", "An echo from beyond... ")
        }
    }

    private func getChunkPrimer(chunkTitle: String, persona: OraclePersona, config: ReadingConfiguration) -> String {
        if config.useVisions {
            return "\n\n"
        } else {
            let title = chunkTitle.uppercased()
            switch persona {
            case .seer: return "\n\n**Portents of \(title)**\n"
            case .peasant: return "\n\n**Signs in the \(title)**\n"
            case .count: return "\n\n**Observations on the \(title)**\n"
            case .spirit: return "\n\n**Echoes from the \(title)**\n"
            }
        }
    }

    private func sample(logits: MLXArray, temperature: Float, repetitionPenalty: Float, pastTokenIDs: [Int]) -> Int {
        var modifiedLogits = logits
        if repetitionPenalty != 1.0 && !pastTokenIDs.isEmpty {
            let uniqueIndices = Set(pastTokenIDs).map { Int32($0) }
            let indices = MLXArray(uniqueIndices)
            let appearedLogits = modifiedLogits[indices]
            let penalized = MLX.where(appearedLogits .> 0, appearedLogits / repetitionPenalty, appearedLogits * repetitionPenalty)
            modifiedLogits[indices] = penalized
        }
        return MLXRandom.categorical(modifiedLogits / temperature).item(Int.self)
    }
}
