
# PlanLens - AI-Powered 401(k) Plan Document Q&A System

## Overview

PlanLens is a Retrieval-Augmented Generation (RAG) system that answers questions about 401(k) plan documents with **verifiable citations**.

## Key Features

- **Citation Enforcement**: Every answer cites specific page numbers
- **Document Grounding**: Answers ONLY from provided plan document
- **Comprehensive Evaluation**: 20-question test set across all plan areas
- **ChatGPT Comparison**: Quantified improvement over generic LLMs
- **Table Handling**: Preserves vesting schedules and contribution formulas

## Architecture

```
PDF Document
    ↓
[1] Ingestion: pdfplumber → section-aware chunks
    ↓
[2] Embedding: sentence-transformers → ChromaDB
    ↓
[3] Retrieval: Question → semantic search → top-5 chunks
    ↓
[4] Generation: Groq (Llama 3.3 70B) → Cited answer
```

## Tech Stack

- **PDF Parsing**: pdfplumber
- **Orchestration**: LlamaIndex
- **Embeddings**: sentence-transformers (local, free)
- **Vector Store**: ChromaDB (local, free)
- **LLM**: Llama 3.3 70B via Groq API (free tier)
- **Evaluation**: Enhanced manual metrics
- **UI**: Gradio

## Installation

```bash
pip install pdfplumber llama-index llama-index-embeddings-huggingface
pip install llama-index-llms-groq llama-index-vector-stores-chroma chromadb
pip install sentence-transformers gradio pandas
```

## Usage

1. Get free Groq API key: https://console.groq.com/keys
2. Run notebook: `PlanLens_Final_Complete.ipynb`
3. Upload your 401(k) plan PDF
4. Ask questions!

## Evaluation Results

**Comprehensive Test (20 questions):**
- **Faithfulness**: 0.95 (Target: ≥ 0.90) ✅
- **Relevancy**: 1.00 (Target: ≥ 0.85) ✅
- **Citation Rate**: 100% (18/18 regular questions) ✅
- **Adversarial Refusal**: 100% (2/2 questions) ✅

**ChatGPT Comparison:**
- Specificity: +400% improvement
- Verifiability: +∞ (0% → 100% citations)
- Actionability: +150% improvement
- Accuracy: +67% improvement

## Author

Vanshi Patel | Northeastern University | INFO 7375
Software Engineering Co-op @ PlanSync (401(k) fintech)
