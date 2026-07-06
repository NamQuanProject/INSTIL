# INSTIL

**Instruction-Anchored Continual Learning** — the natural-language instruction
that defines each task predicts weight-space gradient conflict, so instruction
similarity can *certify* the sign of transfer (backward & forward) without replay
or a trainable router.

* Proposal: [`ideas/instil.pdf`](ideas/instil.pdf)
* Implementation, data prep, and full train/test pipeline: **[`README_INSTIL.md`](README_INSTIL.md)**
* Reference infrastructure (data, metrics, LoRA conventions): [`SAPT/`](SAPT/)

## Quick start

```bash
pip install -r requirements.txt

# 1) get the SuperNI data (SAPT/ is git-ignored, so a fresh clone has none)
python scripts/prepare_superni_data.py

# 2) CPU self-check of the method — no downloads
python scripts/demo_synthetic.py

# 3) full continual train + test with metrics (BWT / FWT / Forgetting / OP)
python scripts/run_instil_t5.py --help
```

See [`README_INSTIL.md`](README_INSTIL.md) for the task orders, the Law-validation
experiment, the config reference, and how each of the paper's guarantees maps to
the code.
