# Architecture Notes

Key concepts to understand before and during fine-tuning.

---

## LoRA in 30 Seconds

Instead of updating all weights W, LoRA learns a low-rank update: **W' = W + BA**

Where B and A are small matrices. If W is 4096×4096 (16M params) and rank r=16:
- A: 4096×16 = 65K
- B: 16×4096 = 65K  
- **Total: 130K trainable vs 16M frozen = 99% reduction**

Key hyperparameters:
- `r` (rank): Higher = more capacity, more memory. Start with 16.
- `alpha`: Scaling factor. Usually 2× rank.
- `target_modules`: Which layers to adapt (attention projections).

---

## VLM Architecture in 30 Seconds

```
Image → Vision Encoder → visual tokens → Projection → LLM input space
                                                            ↓
Text  → Tokenizer → text tokens ──────────────────→ Concatenate
                                                            ↓
                                                       LLM Backbone
                                                            ↓
                                                       Output
```

**Why freeze vision encoder?** It already sees well (trained on millions of images). We're teaching the LLM backbone to *reason* about what it sees — that's where domain knowledge lives.

---

## Evaluation: Three Questions

1. **Did it improve?** — Quantitative metrics
2. **Why did it improve?** — Error analysis
3. **Is it meaningful?** — LLM-as-Judge
