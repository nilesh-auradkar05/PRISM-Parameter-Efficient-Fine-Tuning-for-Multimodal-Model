# PRISM

### Parameter-efficient Reasoning for Industry-Specific Multimodal Systems

> **Teaching small models to think like domain experts — one adapter at a time.**

PRISM is a systematic exploration of parameter-efficient fine-tuning across healthcare and finance, progressing from text classification through instruction tuning to multimodal vision-language models. Every experiment follows an evaluation-driven methodology: baseline first, fine-tune second, measure everything third.

---

## Roadmap

| Phase | Notebook | What It Teaches | Status |
|-------|----------|----------------|--------|
| 0 | `00_environment_setup.ipynb` | Verify Unsloth, PEFT, DeepEval work | ✅ |
| 1 | `01_financial_sentiment.ipynb` | Complete fine-tuning loop on classification | ✅ |
| 2 | `02_medical_chat.ipynb` | Instruction tuning, chat templates, generative eval | ✅ |
| 3 | `03_financial_qa.ipynb` | Context-grounded QA, faithfulness, hallucination detection | ✅ |
| 4 | `04_vlm_deep_dive.ipynb` | VLM architecture exploration (no training) | ✅ |
| 5 | `05_medical_vqa.ipynb` | Fine-tune VLM on radiology images | ✅ |
| 6 | `06_document_vqa.ipynb` | Fine-tune VLM on business documents (DocVQA) | ✅ |
| 7 | `07_portfolio.ipynb` | Demo app, final evaluation | 🔜 |

---

## Completed Work

### Phase 1 · Financial Sentiment Classification ✅

| | |
|---|---|
| **Task** | Classify financial news headlines as positive, negative, or neutral |
| **Dataset** | `financial_phrasebank` (4,846 sentences labeled by domain experts) |
| **Model** | Qwen3-4B-Instruct (8-bit quantized) |
| **Method** | Generative classification — model outputs the label as text |
| **Training** | Full dataset, 5 epochs, batch size 8, cosine LR schedule |
| **Evaluation** | Accuracy, weighted F1, per-class precision/recall, confusion matrix |

**Why generative classification matters:** Instead of adding a classification head, we teach the LLM to generate the label as text. The same SFT pipeline works for single-word labels ("positive"), short answers, and multi-paragraph explanations — no architecture changes needed. This is the foundation every subsequent phase builds on.

**Skills demonstrated:** End-to-end SFT loop, LoRA configuration, DeepEval integration, baseline vs. fine-tuned comparison, error analysis.

### Phase 2 · Medical Instruction Tuning ✅

| | |
|---|---|
| **Task** | Answer medical questions in natural language |
| **Dataset** | `medalpaca/medical_meadow_medical_flashcards` (human-curated medical Q&A) |
| **Model** | Llama3.2-3B-Instruct (4-bit quantized) |
| **Method** | Chat-format SFT with instruction-response pairs |
| **Training** | 4,000 examples, 5 epochs, batch size 8, cosine LR |
| **Evaluation** | ROUGE-1/2/L, keyword coverage, DeepEval GEval, AnswerRelevancyMetric |

**Why this phase matters — the evaluation problem:** When the model generates "Pneumonia presents as persistent cough with fever" and the reference says "Common symptoms include cough, high temperature, and difficulty breathing," exact match scores zero. But the answer is correct. This is where the three-tier evaluation framework becomes essential: ROUGE for quick iteration, keyword coverage for domain-critical terms, and LLM-as-Judge (GEval) for semantic correctness.

**Skills demonstrated:** Instruction tuning, generative evaluation (ROUGE + LLM-as-Judge), answer relevancy assessment, keyword coverage metrics, systematic error analysis.

### Phase 3 · Financial QA with Context Grounding ✅

| | |
|---|---|
| **Task** | Answer financial questions grounded in a provided passage |
| **Dataset** | `sujet-ai/Sujet-Finance-Instruct-177k` (QA with Context subset, ~40k examples from annual reports and filings) |
| **Model** | MistralAI.7B-Instruct (4-bit quantized) |
| **Method** | Chat-format SFT with context passage as user input |
| **Training** | 4,000 examples, 3 epochs, batch size 3 × 8 gradient accumulation, cosine LR |
| **Evaluation** | ROUGE-1/2/L, grounding score, DeepEval FaithfulnessMetric, GEval (Financial Correctness) |

**Why this phase matters — the hallucination problem:** Phase 1-2 tested what the model memorized during training (parametric knowledge). Phase 3 tests whether the model can answer from text given at inference time (contextual knowledge). This is the skill that powers RAG systems, document QA, and report analysis. The new failure mode is hallucination: the model generates a plausible answer that isn't actually supported by the context. Detecting this requires a fundamentally different evaluation — faithfulness — which checks whether every claim can be traced back to the source passage.

**Key technical decisions:**
- `max_seq_length` increased to 1536 to accommodate context passages + questions + answers.
- Batch size reduced to 3 (from 6) due to longer sequences, compensated with 8× gradient accumulation.
- Custom grounding score heuristic provides a free, fast proxy for faithfulness before expensive API evaluation.

**Skills demonstrated:** Context-grounded QA, faithfulness evaluation, hallucination detection, DeepEval FaithfulnessMetric, grounding heuristics, dataset filtering by task type.

### Phase 4 · VLM Architecture Deep-Dive ✅

| | |
|---|---|
| **Task** | Understand Vision-Language Model internals before fine-tuning |
| **Model** | Qwen3-VL-8B-Instruct (4-bit quantized) |
| **Method** | Exploratory — load, inspect, trace data flow, run inference |
| **Training** | None (study phase, ~2-3 credits) |
| **Deliverable** | Interactive architecture diagram + annotated code walkthrough |

**Why a study phase matters:** VLMs have three distinct components: a vision encoder (ViT/SigLIP) that extracts visual features, a projection layer (MLP) that translates visual embeddings into the LLM's space, and the LLM backbone that reasons over both modalities. Understanding where parameters live and where LoRA adapters go is critical before committing compute to training.

**Key insight:** VLM fine-tuning is asymmetric. The vision encoder is frozen — it already extracts excellent features. LoRA adapters go on the LLM backbone, teaching it to reason about what the vision encoder sees.

**Skills demonstrated:** VLM architecture analysis, component inspection, parameter census, multi-modal inference pipeline, LoRA placement strategy for vision models.

### Phase 5 · Medical Visual Question Answering 🔄

| | |
|---|---|
| **Task** | Answer clinical questions about radiology images |
| **Dataset** | `flaviagiammarino/vqa-rad` (2,248 clinician-curated Q&A pairs on 315 radiology images, CC0 license) |
| **Model** | Qwen3-VL-8B-Instruct (4-bit quantized) |
| **Method** | Vision SFT with LoRA on LLM backbone, vision encoder frozen |
| **Training** | Full train set, 3 epochs, batch size 4 × 6 gradient accumulation, cosine LR |
| **Evaluation** | CLOSED accuracy (yes/no), OPEN token F1 + exact match, DeepEval GEval (Medical VQA Correctness) |

**Why this phase matters — the multimodal leap:** Phases 1-3 processed text only. Phase 5 combines everything: LoRA training (Phase 1), generative evaluation (Phase 2), domain-specific metrics (Phase 3), and VLM architecture understanding (Phase 4). The model must look at an X-ray, CT scan, or MRI and answer clinical questions about what it observes.

**Key technical decisions:**
- `finetune_vision_layers=False` keeps the ViT frozen — teaching reasoning, not perception.
- `UnslothVisionDataCollator` handles image preprocessing automatically.
- Dual evaluation tracks: CLOSED (yes/no accuracy) and OPEN (token F1 + GEval semantic correctness).
- VLM-specific SFTConfig flags: `remove_unused_columns=False`, `dataset_kwargs={"skip_prepare_dataset": True}`.

**Skills demonstrated:** VLM fine-tuning, multi-modal data pipeline, UnslothVisionDataCollator, dual-track VQA evaluation, domain-specific GEval criteria, medical image analysis.

### Phase 6 · Document Visual Question Answering 🔄

| | |
|---|---|
| **Task** | Answer questions about scanned business documents (memos, invoices, reports) |
| **Dataset** | `nielsr/docvqa_1200_examples` (1,200 Q&A pairs on industry documents from UCSF library) |
| **Model** | Qwen3-VL-8B-Instruct (4-bit quantized, fresh LoRA) |
| **Method** | Vision SFT with LoRA on LLM backbone, vision encoder frozen |
| **Training** | 1,000 examples, 3 epochs, batch size 8 × 8 gradient accumulation, cosine LR |
| **Evaluation** | ANLS (Average Normalized Levenshtein Similarity), exact match, DeepEval GEval (Document Extraction Correctness) |

**Why this phase matters — cross-domain transfer:** Phase 5 fine-tuned a VLM on medical images. Phase 6 applies the exact same approach to business documents — same model, same LoRA config, same training code. The only changes are the dataset and system prompt. This proves the VLM fine-tuning methodology generalizes. DocVQA demands fundamentally different visual skills: OCR-level text reading and layout understanding rather than clinical pattern recognition.

**Key technical decisions:**
- ANLS metric replaces exact match as the primary metric — handles OCR noise, formatting variations ("Jan" vs "January"), and multiple valid answer representations.
- Answer categorization (numeric, date, short_text, long_text) for granular error analysis.
- Cross-domain comparison framework to contrast medical VQA vs. document VQA performance.

**Skills demonstrated:** Document VQA, ANLS evaluation metric, cross-domain transfer analysis, answer categorization for error analysis, enterprise document understanding.

---

## Evaluation Framework

**Tier 1 — Automated Metrics**: Accuracy, weighted F1, per-class precision/recall for classification. Token F1 and exact match for VQA. ANLS (Average Normalized Levenshtein Similarity) for document extraction.

**Tier 2 — Overlap & Coverage**: ROUGE-1/2/L, keyword coverage, grounding score heuristic.

**Tier 3 — LLM-as-Judge**: DeepEval GEval (semantic correctness), AnswerRelevancyMetric, FaithfulnessMetric, Medical VQA Correctness.

---

## Core Concepts

**LoRA (Low-Rank Adaptation):** Learn low-rank weight updates W' = W + BA. For rank r=16 on a 4096×4096 matrix, trainable parameters drop from 16M to 130K — a 99% reduction.

**Generative Classification:** Teach the model to generate labels as text. Same SFT pipeline works for all output types — no architecture changes needed.

**LLM-as-Judge:** GPT-4o-mini via DeepEval evaluates semantic correctness when exact match fails.

**Faithfulness Evaluation:** Decompose answers into claims, verify each against source context. Catches subtle hallucinations word overlap misses.

**VLM Architecture:** Vision encoder (frozen ViT) → projection layer (MLP) → LLM backbone (LoRA targets). Images become token sequences concatenated with text.

---

## Project Structure

```
prism/
├── docs/
│   └── architecture_notes.md         # LoRA + VLM concepts documented
├── notebooks/
│   ├── environment_setup.ipynb     # Phase 0—verify tooling
│   ├── financial_sentiment.ipynb   # Phase 1—classification
│   ├── medical_chat.ipynb          # Phase 2—instruction tuning
│   ├── financial_qa.ipynb          # Phase 3—context-grounded QA
│   ├── vlm_deep_dive.ipynb         # Phase 4—VLM arch study
│   ├── medical_vqa.ipynb           # Phase 5—medical VQA FT
│   └── document_vqa.ipynb          # Phase 6—document VQA FT
├── pyproject.toml
└── README.md
└── Project_Overview.jsx
```

---

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Fine-tuning | Unsloth | 4× faster training, 60% less memory |
| PEFT | LoRA via PEFT library | Parameter-efficient adaptation |
| Evaluation | DeepEval | GEval, FaithfulnessMetric, AnswerRelevancy |
| Text Models | Qwen2.5-1.5B-Instruct | Phases 1-3 |
| Vision Model | Qwen2.5-VL-3B-Instruct | Phases 4-6 |
| Experiment Tracking | MLflow + Databricks | Reproducibility and model management |

---

## Quick Start

```bash
# Open any notebook in Google Colab
# Select a GPU runtime (T4 works, A100 is faster)
# Run cells sequentially — each notebook is a complete experiment
```
