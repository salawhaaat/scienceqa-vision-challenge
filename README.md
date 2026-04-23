# ScienceQA Vision Challenge

Fine-tuning `SmolVLM-500M-Instruct` with QLoRA to solve visual multiple-choice science questions (Geography, Biology, Physics).


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salawhaaat/scienceqa-vision-challenge/blob/main/notebook-qlora.ipynb)

**Dataset (verified):** 3,109 train | 1,048 val | 1,008 test | 5,165 images | 0 missing

---

## Method

| | |
|---|---|
| **Model** | `HuggingFaceTB/SmolVLM-500M-Instruct` (~500M params) |
| **Fine-tuning** | QLoRA — 4-bit NF4 + LoRA adapters (~3-5M trainable params) |
| **Scoring** | Log-likelihood over answer choices |
| **GPU** | T4x2 16 GB |
| **Storage** | Kaggle Output tab — download adapter zip after each run |


---

## Decision Rules

```
quick_val < 35%  → skip full eval, fix config   (random baseline = 33%)
quick_val ≥ 35%  → run full val, compare with best
full_val > best  → new best, download adapter + submit
full_val ≤ best  → discard, try next config
```

---

## Experiment Plan

### 🔴 Phase 1 — Core (runs 01-05, do all of these)

| Run | What changes | Config | Est. impact | Quick Val | Full Val | Time | Status |
|-----|-------------|--------|-------------|-----------|----------|------|--------|
| 01 | Baseline | `r=16, attn only, lr=2e-4, epochs=3` | reference | — | — | — | 🔲 |
| 02 | +MLP LoRA layers | `targets += gate/up/down_proj, r=8` | +3-5% | — | — | — | 🔲 |
| 03 | LR down | `lr=1e-4` on best of 01-02 | stability | — | — | — | 🔲 |
| 04 | LR up | `lr=5e-4` on best of 01-02 | faster convergence | — | — | — | 🔲 |
| 05 | Scoring fix | normalize log-likelihood by answer token only | cleaner signal | — | — | — | 🔲 |

### 🟡 Phase 2 — Refinement (runs 06-08, only if compute allows)

| Run | What changes | Config | Est. impact | Quick Val | Full Val | Time | Status |
|-----|-------------|--------|-------------|-----------|----------|------|--------|
| 06 | Epochs sweep | `epochs=2` then `epochs=4` on best config | overfitting check | — | — | — | 🔲 |
| 07 | Dropout | `dropout=0.1` on best config | regularization | — | — | — | 🔲 |
| 08 | Higher resolution | `img_size=384` on best config | diagram detail | — | — | — | 🔲 |

### 🟢 Phase 3 — Polish (run 09, if in top 3)

| Run | What changes | Config | Est. impact | Quick Val | Full Val | Time | Status |
|-----|-------------|--------|-------------|-----------|----------|------|--------|
| 09 | Ensemble | avg logits epoch 2 + epoch 3 of best run | free +1-2% | — | — | — | 🔲 |

---

## Results Summary

| Run | Full Val Acc | Δ vs best | New best? | Training time | Key takeaway |
|-----|-------------|-----------|-----------|---------------|--------------|
| 01  | —           | —         | —         | —             | — |

---

## Baseline Config

```python
MODEL_ID      = "HuggingFaceTB/SmolVLM-500M-Instruct"
LORA_TARGETS  = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 3
BATCH_SIZE    = 4
GRAD_ACCUM    = 4        # effective batch = 16
IMG_SIZE      = 224
MAX_LENGTH    = 512
QUICK_VAL_N   = 150
```