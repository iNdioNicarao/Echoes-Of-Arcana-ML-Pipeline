# Echoes-Of-Arcana-ML-Pipeline

**Note: The front-end application code and core mechanics of Echoes of Arcana are closed-source and in active commercial development. This repository serves strictly as a technical breakdown of the underlying QWEN fine-tuning pipeline and ML architecture.**

## Overview

This repository showcases the Machine Learning pipeline built for "Echoes of Arcana," a robust application integrating localized, on-device AI inference via MLX. The purpose of this pipeline is to train a highly specialized and deeply evocative "Voice of the Oracle." 

Rather than relying entirely on heavy prompt engineering, this pipeline utilizes LoRA (Low-Rank Adaptation) fine-tuning to physically bake esoteric vocabulary, natural card synthesis, and archetypal behavior into the base model.

## ML Architecture & Fine-Tuning Strategy

### Base Model
We utilized the **Qwen2.5-1.5B-Instruct** model, chosen for its efficiency and strong base instruction-following capabilities, optimizing it for local on-device execution using MLX on Apple Silicon. 

### LoRA Fine-Tuning
The fine-tuning process employs a Low-Rank Adaptation (LoRA) approach. This technique isolates the updates to a small set of auxiliary weights, making the training process highly efficient while preventing catastrophic forgetting of the model's foundational linguistic skills.

**Key Objectives:**
- Teach the model to generate non-robotic, sweeping metaphorical transitions.
- Enforce strict structural flows in readings (e.g., Opening -> Synthesis -> Closing).
- Embed deep esoteric concepts such as Astrological integration (Sun/Moon/Rising signs) and Numerological pattern recognition.

### Dataset Structure
The custom dataset was programmatically constructed to simulate complex esoteric reasoning. 
- **Size:** 1,000 high-quality instructional examples.
- **Composition:** 
  - 700 Core Readings focusing on Tarot card meanings and interactions.
  - 100 Readings emphasizing Astrological synthesis.
  - 100 Readings tracking Numerological echoes and elemental clashes.
  - 100 Readings simulating journaling and follow-ups.

*See `dataset_sample.jsonl` for a sanitized sample of the data structures used.*

### Dynamic Archetypes
To ensure longevity and variation, the fine-tuning data introduced conditional archetype triggers (e.g., "The Trickster" or "The Shadow Weaver"). By blending these examples, the model learns to shift its tone based on the emotional weight of the drawn cards or the time of day.

## Contents
- `generate_dataset.py`: The python script responsible for algorithmically constructing the esoteric training data.
- `diagnostics/`: MLX diagnostic scripts used for verifying model integrity, tokenization, RoPE positioning, and inference testing.
- `dataset_sample.jsonl`: Sanitized proof-of-work dataset samples demonstrating the input-output structural alignments.

## Proof of Inference (Offline/On-Device)
*Demonstrating the LoRA fine-tuned Qwen model running locally on Apple Silicon, completely decoupled from external APIs.*

![Offline Inference Demo](EdgeAI.gif)
---
*Developed by Dennis Isaac Gutierrez Zeledon as a showcase of ML capabilities and fine-tuning expertise.*