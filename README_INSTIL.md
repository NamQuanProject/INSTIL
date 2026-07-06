# Instil — Instruction-Anchored Continual Learning

Reference implementation of the proposal in [`ideas/instil.pdf`](ideas/instil.pdf):

> **The instruction that defines each task predicts weight-space gradient conflict.**
> Instil splits every task's low-rank update into a **null-space** part (provably
> non-interfering) and an **occupied-space** part admitted *in proportion to
> instruction similarity*. This turns subspace *isolation* (O-LoRA / GPM /
> InfLoRA) into *certified transfer*: non-negative backward transfer, forward
> transfer, exact non-forgetting, and training-free routing / zero-shot
> composition — with no replay and no trainable router.

This package is built on the same stack as the bundled **SAPT** codebase
(PyTorch + Transformers, LoRA on the attention `q, v` projections, the SuperNI
CIT benchmark under `data/`) and reuses SAPT's metric definitions so
numbers are directly comparable.

---

## 1. What's here

```
instil/                     the library
  config.py                 InstilConfig — all hyper-parameters                [§5,§8]
  encoders.py               frozen instruction encoder E -> unit prototypes    [§3,§8]
  gate.py                   instruction gate  gamma = sigma(a<p_t,p_j> + b)    [§5.2]
  subspace.py               GPM occupied-subspace bookkeeping U_j, P^perp      [§5.1]
  lora.py                   InstilLoRALinear (SAPT-style A/B) + injection       [§5,§8]
  update.py                 gate hook (structural — no-op here)                [§5.2]
  instil.py                 orchestrator: LearnTask / Answer, Merge & Bank      [§5.3,§7]
  metrics.py                OP / Forgetting / BWT / FWT (matches score.py)      [§9]
  law.py                    Experiment 1 — validate the Instruction–Gradient Law[§4,§9]
  trainer.py                thin continual-training loop                        [§7]
  data_superni.py           SuperNI / Long-Sequence loaders (data/)
  textscore.py              ROUGE-L / exact-match scorers (no external deps)

scripts/
  prepare_superni_data.py   (re)build data/ from natural-instructions
  run_instil_t5.py          FULL train + test pipeline on SuperNI with T5
  run_law_experiment.py     Experiment 1 driver (the "plot that decides the paper")
  demo_synthetic.py         CPU-only end-to-end demo & self-check (no downloads)

tests/test_core.py          optional unit tests for the structural guarantees
requirements.txt
```

## 2. Install

The library is decoupled from SAPT's frozen `requirements.txt`; a recent
PyTorch + Transformers is enough:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2b. Get the data

The SuperNI CIT data lives under `data/SuperNI/<task>/{train,dev,test}.json`.
It ships with the repo, so **if `data/` is already populated you can skip this
section.** To (re)build it from scratch on a new machine, or to regenerate
different split sizes, run the dependency-free preparer — each split file is just
the authoritative Natural-Instructions task file with its `Instances` sliced:

```bash
# downloads the 15 standard SuperNI CIT tasks into data/SuperNI
python scripts/prepare_superni_data.py
```

Offline / behind a proxy? Clone the source and point at it:

```bash
git clone --depth 1 https://github.com/allenai/natural-instructions
python scripts/prepare_superni_data.py --source natural-instructions/tasks
```

Options: `--tasks a,b,c` (subset), `--train_size/--dev_size/--test_size`,
`--output_dir` (default `data`), `--force` (rebuild), `--no_shuffle`. Existing
files are kept unless `--force`. Splits are disjoint and reproducible (seeded);
they are valid and comparable across runs even if not byte-identical to the
authors' original indices.

## 3. Smoke test (no model downloads, CPU)

Verifies the core mechanics end-to-end on synthetic tasks — exact
non-forgetting (Prop. 1), the gate matrix, routing (Prop. 2), and zero-shot
composition:

```bash
python scripts/demo_synthetic.py
# ... -> "ALL CHECKS PASSED"
```

## 4. Full pipeline: train **and** test on SuperNI

`scripts/run_instil_t5.py` streams the task order, learns each task with the
instruction-gated update, and **after every task evaluates all seen tasks** to
fill the lower-triangular result matrix `R`, from which OP / Forgetting / BWT /
FWT are computed (§9). Everything (training, evaluation, metrics, checkpoint)
happens in one process. (Run `scripts/prepare_superni_data.py` first if
`data/` is empty — see §2b.)

```bash
python scripts/run_instil_t5.py \
  --model_name_or_path t5-large \
  --data_dir data --benchmark SuperNI \
  --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
  --mode bank --lora_r 8 --lora_alpha 16 --target_modules q,v \
  --epochs 5 --lr 3e-4 --batch_size 8 \
  --output_dir logs_and_outputs/instil_superni_order1
```

**Quick CPU sanity run** (tiny model, few examples):

```bash
python scripts/run_instil_t5.py \
  --model_name_or_path t5-small \
  --data_dir data \
  --task_order task363_sst2_polarity_classification,task1687_sentiment140_classification,task875_emotion_classification \
  --mode bank --epochs 1 --max_train 200 --max_eval 50 --device cpu \
  --output_dir logs_and_outputs/instil_smoke
```

Outputs in `--output_dir`:

* `results.json` — the task order, the full `R` matrix, and the metrics.
* `instil_state.pt` — prototypes + per-layer adapters / bank / subspaces.
* `task_order.txt` — for compatibility with SAPT's `score.py` layout.

Printed at the end:

```json
{ "OP": ..., "Forgetting": ..., "BWT": ..., "FWT": ... }
```

The headline the paper targets is **positive BWT and FWT at equal-or-better OP**
— the regime isolation methods (BWT = FWT = 0) cannot enter.

### Task orders (from SAPT `gen_script_superni_t5.py`)

* **Order 1** — the comma-separated list used above.
* **Order 2** — reverse-ish permutation; see `SAPT/gen_script_superni_t5.py`
  (`order_idx = 2`). Robustness to order is an ablation in §9.

### Metric (choose per benchmark, matching SAPT)

* SuperNI → `--metric rougeL` (default).
* Long-Sequence → `--metric exact_match` and `--benchmark Long_Sequence`.

## 5. Experiment 1 — validate the Law (run this first; it's cheap)

Scatters instruction similarity `<p_t,p_j>` against measured subspace gradient
alignment `<∇L_j, ∇L_t>_{U_j}` over task pairs, and reports Pearson `r`, sign
accuracy, and the fitted zero-crossing `rho0` (which the gate then consumes):

```bash
python scripts/run_law_experiment.py \
  --model_name_or_path t5-large \
  --data_dir data \
  --task_order <same comma-separated order as above> \
  --max_batches 4 --output_dir law_out
```

Produces `law_out/law_points.csv`, `law_out/law_summary.json`, and (if
matplotlib is installed) `law_out/law_scatter.png`. A positive `pearson` with
`sign_accuracy` well above chance is the empirical crux of the paper. You can
then pass the fitted value as `--rho0` to `run_instil_t5.py`.

## 6. How the paper's guarantees map to the code

| Paper | Where | Mechanism in code |
|---|---|---|
| Eq. (1) gated update | `lora.py` `set_adapter_basis` + `instil.py` `_build_basis` | frozen basis `A = [free dirs \| gamma_{t,j}·U_j]`; only `B` trains |
| Prop. 1 non-forgetting | `subspace.py` `free_directions`, `lora.py` | free rows ⟂ `span(U_<t)` ⇒ `dW·x = 0` on prior span, for **any** `B` (verified in the demo) |
| Thm. 1 +BWT / Cor. 1 | gate admits occupied block `j` iff `<p_t,p_j> ≥ rho0` | `gate.py` `InstructionGate`; `gamma=0` recovers isolation |
| Thm. 2 FWT warm-start | `instil.py` `_warm_start` | init `B` from nearest prior adapter via least squares |
| Prop. 2 routing | `instil.py` `routing_weights`, `answer` | frozen nearest-instruction lookup — zero trainable routing params |
| §5.4 composition | `lora.py` bank + `answer` | `dW* = Σ_t w_t·dW_t` for blended instructions |
| §5.1 bookkeeping | `subspace.py` `add_task_cov` | streaming covariance `XᵀX`, top-`r` eigenvectors via randomized subspace iteration (no full SVD), energy ≥ 0.95, orthogonalised |
| §9 metrics | `metrics.py` | same formulas as SAPT `score.py` |

## 7. Modes (`--mode`)

* **`bank`** (default) — stores a tiny per-task adapter; inference routes/soft-composes
  over the bank (Alg. 2). Non-forgetting is by construction (separate adapters +
  frozen routing) and you additionally get zero-shot composition.
* **`merge`** — one shared adapter folded in place; memory does not grow. This is
  where the **provable** non-forgetting floor and certified BWT live (the demo's
  check 1). Answers use the single shared model with the instruction in context.

## 8. Using Instil in your own loop

```python
from instil import InstilConfig, Instil, inject_instil_lora, MeanPooledBackboneEncoder
from instil.trainer import ContinualTrainer

cfg = InstilConfig(lora_r=8, lora_alpha=16, target_modules=["q", "v"], mode="bank")
inject_instil_lora(model, cfg)                      # wraps q,v Linears; freezes base
encoder = MeanPooledBackboneEncoder(model, tokenizer)
instil  = Instil(model, encoder, cfg)

trainer = ContinualTrainer(instil, loss_fn=lambda m, b: m(**b).loss, epochs=5)
for instruction, train_loader in stream:
    trainer.learn_task(instruction, train_loader)   # Alg. 1

with instil.answer(query_instruction):              # Alg. 2 (bank routing)
    out = model.generate(**inputs)
```

If you drive training yourself (e.g. a HuggingFace `Trainer`), just call
`instil.project_gradients()` after each `loss.backward()` — it is a safe no-op in
this build (the gate is enforced structurally by the frozen basis `A`).

## 9. Relationship to SAPT

* **Data / metrics**: same as SAPT — SuperNI tasks under `data/`, RougeL/EM, and the
  BWT/FWT/Forgetting definitions in `SAPT/score.py`.
* **Adapter**: same LoRA parameterisation (`A: r×in`, `B: out×r`, `q,v`).
* **Difference**: SAPT is *learned selection* (a trainable shared-attention
  selector + generative pseudo-replay). Instil is **replay-free and
  router-free**: the instruction gate and nearest-instruction routing are frozen
  and training-free, and the transfer sign is *certified* rather than empirical.
  In the capability matrix (Table 1) Instil fills the empty `+BWT ∧ +FWT ∧
  replay-free ∧ router-free ∧ guaranteed` cell.

## 10. Implementation notes & knobs

* **Encoder `E`** — default is the mean-pooled last-hidden-state of the
  instruction through the *frozen* backbone (`MeanPooledBackboneEncoder`, run
  with adapters bypassed so prototypes stay drift-free). Swap in a sentence
  encoder to ablate "encoder choice" (§9); a dependency-free `HashingEncoder` is
  provided for CPU tests/demo.
* **Gate** — `--gate_slope_a` sharpens reinforcement; `--rho0` fixes the
  zero-crossing (`b = -a·rho0`). Omit `--rho0` to default to `0.0`, or fit it via
  Experiment 1 and pass it in. `gamma < gate_floor` is clamped to exactly `0`
  so the non-forgetting guarantee is numerically exact.
* **Subspace** — `energy_threshold` (default 0.95) and `subspace_rank_cap` bound
  each task's stored basis; `max_activation_samples` (~2k) caps the covariance
  collection budget. Bases are tiny (`in × R`) and kept on CPU.
* **Storage** — merge: only `{U_j}` + `{p_j}` (MBs). bank: additionally the
  per-task LoRA deltas.
* **Progress & logging** — training/eval show live `tqdm` bars (per-task loss,
  collection pass, generation), and every run is logged to
  `logs/<run_name>.log` (both stdout and file). `logs/` and `logs_and_outputs/`
  are git-ignored. Control with `--log_dir` / `--run_name`; `tqdm` degrades to a
  no-op if not installed, so nothing breaks without it.

### Computational efficiency (the expensive `torch.linalg.svd` is gone)

The GPM bookkeeping used to run a full `torch.linalg.svd` on the whole
`N × in` activation matrix of every tracked layer every task — the dominant
cost (LAPACK, largely CPU-bound, `O(N·in²)`), plus `O(N·in)` memory to store the
activations. Following HESTIA's online-statistics idea
(`HESTIA/lib/signatures.py`: `OnlineDiagonalGMM` streams Welford means/vars
instead of hoarding features), Instil now:

1. **Streams a fixed-size covariance** `C = XᵀX` (`in × in`) during the
   collection pass (`InstilLoRALinear._maybe_capture`) — never materialising the
   `N × in` matrix. Memory drops from `O(N·in)` to `O(in²)`. Collection also
   **stops early** once every layer has its row budget (`max_activation_samples`),
   so it never sweeps the whole train set.
2. **Extracts only the top-`r` eigenvectors** of the small symmetric `C` with
   **randomized subspace iteration** (Halko–Martinsson–Tropp — the method behind
   scikit-learn's `randomized_svd`): a random probe, a couple of power iterations,
   and one tiny `(r+p)×(r+p)` eigh. This is `O(in²·r)`, all dense matmul/QR, runs
   **on the GPU**, and — unlike the earlier `torch.lobpcg` — has **no iterative
   convergence loop that can stall** (that stall was the reported hang). A full
   `eigh` is used only for small matrices where it is already cheap. Eigenvectors
   of `XᵀX` are exactly the right singular vectors of `X`, so the subspace is
   identical; `trace(C)` gives total energy for the 0.95 cutoff.

Net effect: the per-task, per-layer decomposition goes from a full SVD over
thousands of rows (CPU) to a randomized top-`r` eigensolve on a single `in × in`
matrix on the GPU, with a bounded memory footprint — the subspace is unchanged.
Knobs: `SubspaceMemory(oversample=8, subspace_iters=2, dense_threshold=256)`.
(The raw-activation `add_task` / `free_directions` wrappers are kept for
compatibility; they just build `XᵀX` and call the covariance path.)

## 11. Troubleshooting

* *"No InstilLoRALinear layers found"* — call `inject_instil_lora(model, cfg)`
  before constructing `Instil`, and check `--target_modules` matches your
  backbone's linear names (`q,v` for T5/LLaMA attention).
* *Out of memory on `t5-large`* — lower `--batch_size`, `--max_source_length`,
  or use `t5-base`; the demo/smoke commands run on CPU.
* *Prototypes look identical* — the `HashingEncoder` is bag-of-words; for real
  runs use `MeanPooledBackboneEncoder` (the default in the scripts).
* *Hangs right after "Task 0" / subspace step is very slow* — this was the old
  `torch.lobpcg` stalling in `_update_ortho`; it is replaced by GPU randomized
  eig (see §10). If you still see a pause there, it is the one-time
  covariance-collection forward pass; it early-stops at `max_activation_samples`
  rows, so lower that value to shorten it.

## 12. Citation

If you use this, please cite the Instil proposal (`ideas/instil.pdf`) and the
SAPT paper whose infrastructure it builds on:

```bibtex
@inproceedings{zhao2024sapt,
  title={SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of Large Language Models},
  author={Zhao, Weixiang and Wang, Shilong and Hu, Yulin and Zhao, Yanyan and Qin, Bing and Zhang, Xuanyu and Yang, Qing and Xu, Dongliang and Che, Wanxiang},
  booktitle={ACL}, year={2024}
}
```
