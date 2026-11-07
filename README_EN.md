# CodeAgent-R

**English** | [中文](README.md)

A code-repair post-training experiment platform: public task context → teacher candidates → execution grading → isolated data splits → QLoRA SFT → paired base/adapter evaluation.

Real Hugging Face / PEFT single-GPU SFT and checkpoint evaluation scripts are implemented. DPO and GRPO remain offline contracts, losses, and toy validation; complete tool-using agent episodes and online RL are not validated. v6 delivers a directionally consistent test-set gain (see below).

## Results (v6, 2026-11-07)

| Split | Base | SFT | Delta | Improved/Regressed |
|---|---|---|---|---|
| dev (20 tasks, selection) | 10/20 (50%) | 12/20 (60%) | +10.0 pp, monotonic | +3 / −1 |
| **test (20 tasks, frozen)** | **12/20 (60%)** | **13/20 (65%)** | **+5.0 pp, CI [+0, +15]** | **+1 / 0** |

**3-seed replication: mean +1.7pp (0/+5/0), zero regressions; the selection mechanism kept the base model when dev showed no gain.** A directionally consistent, not-yet-significant gain on a 104-task / 52-source-group benchmark (mutation-audited data quality), 202 replay-verified samples, QLoRA 3 epochs, checkpoint selection by dev execution pass@1 (base included). Full numbers, failure analysis, and honest limits: [v6 experiment results](docs/EXPERIMENT-V6-RESULTS.md).

## Evidence and scope

**v6 (2026-11-07): pass@1 60% → 65% on 20 unseen frozen-test tasks (+5 percentage points, 1 improved / 0 regressed, paired bootstrap 95% CI [+0.00, +15.00]).** A directionally consistent, not-yet-significant gain. On the 20-task dev selection split, pass@1 rose monotonically across epochs (50 → 50 → 55 → 60), and every improvement came from hidden boundary-test semantics. Earlier rounds (71/204/60 samples) each had their failure root-caused: insufficient data scale ("style, not semantics"), underdetermined hidden tests (contradictory same-task labels), and 12-task evaluation noise. Full numbers, failure analysis, and honest limits: [v6 experiment results](docs/EXPERIMENT-V6-RESULTS.md).

Key v6 facts: 104 tasks / 52 source groups frozen 64/20/20; teacher candidates migrated across experiments by task fingerprint (56% collection savings); mutation-testing data-quality audit (75 tasks with zero surviving mutants); 202 replay-verified samples / 22,706 supervised tokens; QLoRA 3 epochs / 78 updates; per-epoch immutable snapshots selected by dev execution pass@1 (base included, ties keep base); final evaluation with training- and selection-data contamination checks.

| Component | Implemented | Evidence boundary |
|---|---|---|
| Data / grading | External task JSONL, shared public prompt, source-group splits, duplicate/contamination checks, three-layer grader | Independent tasks still need to be prepared |
| SFT | NF4 QLoRA, full chat-template target masks, token-weighted accumulation, dev selection, adapter save/reload | Target-GPU memory, throughput, and quality require validation |
| Evaluation | Greedy default, protocol/budget checks, full outputs and failure diagnostics, paired intervals | Statistical intervals do not repair leakage or test-set tuning |
| Runtime | Workspaces, snapshots, budget ledger, branches from shared failure states | Process controls are not a validated container sandbox |
| DPO | Real trainer + preference construction + protocol selection (v7: dev 55% did not beat SFT, not deployed per protocol; root cause: teacher-vs-self-failure pairs are already-satisfied preferences with vanishing gradients — see [v7 DPO](docs/EXPERIMENT-V7-DPO.md)) | No hard-negative pairs or iterations yet |
| GRPO | Shared-state contract, GRPO math checks | Not online RL |

## Quick start

Use Linux or WSL2 for Bash/POSIX execution and grading. Python 3.10–3.12 is recommended for GPU training; native Windows is not validated.

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m agentlab doctor
python scripts/prepare_data.py --help
python scripts/train_sft.py --help
python scripts/evaluate.py --help
python scripts/compare.py --help
```

Install CUDA PyTorch compatible with the driver before adding GPU dependencies:

```bash
python -m pip install -e ".[dev,train,api]"
python -m pip check
```

The `train` extra specifies candidate compatibility ranges, not a verified GPU-environment lockfile. Record the actual dependency versions, driver/CUDA, and model revision. `agentlab` currently exposes `doctor`; training and evaluation use `scripts/` directly.

## Reproduction and next steps

The full pipeline (task-pool export and audit → source-group frozen splits → teacher collection with replay verification → training → dev-execution selection → frozen-test final evaluation) is documented in [START-HERE](docs/START-HERE.md) and the [GPU post-training guide](docs/POSTTRAIN-GPU.md); `scripts/run_experiment_v2.sh` runs it stage-by-stage with resume support. Teacher candidates migrate across experiments by task fingerprint (`scripts/migrate_candidates.py`, 56% collection savings in v6). Next: multi-seed replication (`scripts/aggregate_seeds.py`), repeated sampling, and real DPO (P08). **500+ independent training tasks remain a build target; resampling and renames are not independent tasks.**

## Experiment design

The platform separates trusted grading from model-visible context. Packing is disabled initially; target labels come from the full chat template and exclude prompt/padding tokens. Gradient accumulation uses supervised-token counts and includes partial final windows. Training records data, configuration, template, implementation, and model metadata.

Useful controlled comparisons include data quality under equal supervised-token budgets, complete public context for cross-file repairs, successful-answer versus recovery supervision, and state-aligned DPO. Recovery training and real-model DPO require further implementation before those experiments can be claimed.

## Reproducible artifacts

Publish task sources/licenses, split and dataset manifests, training configuration, environment, adapter identity, per-task results, intervals, and failure analysis. The repository currently contains no verified held-out performance gain. Resume wording should reflect implemented engineering until the independent experiments are completed.

See [current status](docs/project-status.yaml). The [2026-11-01 review](docs/REVIEW-FIXES.md) and other older guides are historical records; their missing-script statements, illustrative scores, and fixed costs are not current operating instructions.

## Limitations

- Generated code executes with the current user's permissions; production use requires container, network, and resource isolation.
- Full tool-using episodes, end-to-end time/USD enforcement, real-model DPO, and online RL remain unimplemented or unvalidated.
- Small authored tasks do not establish real-repository repair ability; external licenses, independence, and hidden-test quality need auditing.
- Data, results, run logs, and checkpoints are ignored by Git; publish a separate sanitized reproduction bundle.

Architecture is inspired by [pi](https://github.com/earendil-works/pi)'s minimal harness approach and independently implemented. License: MIT.
