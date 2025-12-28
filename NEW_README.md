# Ambiguity-Aware Sampling for LLaDA-style Masked Diffusion LMs

This repository is a research fork built on top of an **LLaDA-style masked discrete diffusion language model** codebase, adding **ambiguity-aware uncertainty decomposition** for decoding and selective resampling.

Most LLaDA-style decoders use a single softmax-derived confidence (entropy / max-prob / margin) to decide **which masked tokens to update first**. But a single number can mix two qualitatively different situations:

- **Epistemic uncertainty**: the model *doesn’t know yet* (lack of knowledge / insufficient context)
- **Aleatoric (ambiguity) uncertainty**: multiple answers are *intrinsically acceptable* (inherent ambiguity)

We explicitly separate these and use them to design better **unmasking schedules**, **late-commit policies**, and **selective resampling** during diffusion decoding.

> TL;DR: same model checkpoint, smarter decoding.

---

## What’s included

- **Uncertainty decomposition during decoding**
  - Plug-in **MC Dropout** (enable dropout only for uncertainty; baseline remains unchanged)
  - (Optional) **ensemble / multi-checkpoint** mode (if you wire it)
- **Ambiguity-aware scheduling**
  - Commit early where epistemic is low (safe)
  - Delay where epistemic is high (needs more context)
  - Late-commit where aleatoric/ambiguity is high (many acceptable answers)
- **Selective resampling**
  - Resample only positions with persistent high epistemic
  - Preserve diversity for ambiguous positions (avoid over-committing too early)

---

## Method overview

At decoding step \(s\), let \(\Omega_s\) be the set of observed (unmasked) token positions and \(\bar{\Omega}_s\) the masked positions.

For each masked position \(i\in\bar{\Omega}_s\), we obtain \(K\) predictive distributions (e.g., MC Dropout passes):
\[
p_i^{(k)}(\cdot) = p_{\theta_k}(x_i \mid x_{\Omega_s}), \quad k=1\dots K.
\]
Define the mean prediction \(\bar p_i = \frac{1}{K}\sum_{k=1}^K p_i^{(k)}\).

We compute:
- **Total uncertainty**: \(U^{tot}_i = H(\bar p_i)\)
- **Aleatoric uncertainty**: \(U^{ale}_i = \frac{1}{K}\sum_{k=1}^K H(p_i^{(k)})\)
- **Epistemic uncertainty**: \(U^{epi}_i = U^{tot}_i - U^{ale}_i\)

Then we schedule unmasking using a policy such as:
\[
\text{score}_i = -\alpha U^{epi}_i - \beta U^{ale}_i,
\]
and unmask the top \(m_s\) positions by score at step \(s\).

**Intuition**
- High \(U^{epi}\): model disagreement / knowledge gap → **delay** (wait for more context)
- High \(U^{ale}\): intrinsic ambiguity / multiple acceptable tokens → **late-commit** (commit later, keep diversity)

---

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt  # PyTorch, transformers, datasets, etc.
```

### 2. Running GSM8K Benchmark
```bash
# Run a single experiment
./scripts/run_gsm8k.sh

# Run all comparison experiments
./scripts/run_all_benchmarks.sh
```

### 3. Evaluating Results
```bash
# Evaluate all GSM8K results
./scripts/evaluate_all_results.sh

# Compare results
python3 compare_results.py --show_errors --show_common
```

For detailed evaluation guide, see [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

---

## MC Dropout in LLaDA-style decoding

LLaDA-style inference typically runs with `model.eval()` (dropout OFF).  
In this repo, **baseline decoding remains eval-mode**, and we enable dropout **only inside the uncertainty estimator**.

A robust pattern is: keep `model.eval()` globally, but set only `nn.Dropout` modules to train mode:

```python
import torch
import torch.nn as nn

model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.train()

K = 8
with torch.inference_mode():
    probs_sum = 0
    ent_sum = 0
    for _ in range(K):
        logits = model(input_ids, attention_mask=attn).logits  # adapt to your wrapper
        p = logits.softmax(-1)

        probs_sum = probs_sum + p
        ent_sum = ent_sum + (-p * (p.clamp_min(1e-9)).log()).sum(-1)

    p_bar = probs_sum / K
    H_tot = (-p_bar * (p_bar.clamp_min(1e-9)).log()).sum(-1)
    H_ale = ent_sum / K
    H_epi = H_tot - H_ale