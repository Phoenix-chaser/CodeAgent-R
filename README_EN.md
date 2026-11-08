# CodeAgent-R

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-QLoRA_NF4-red)
![transformers](https://img.shields.io/badge/transformers-HuggingFace-yellow)
![tests](https://img.shields.io/badge/tests-470_passed-brightgreen)
![license](https://img.shields.io/badge/license-MIT-green)
![status](https://img.shields.io/badge/experiment-complete-brightgreen)

[中文](README.md) | **English**

A post-training experiment platform for code repair. Applies QLoRA SFT to Qwen2.5-Coder-7B with verified gains on a held-out test set, plus DPO preference optimization (negative result with root-cause analysis).

## Final Results

| Split | Base | SFT | Delta | Improved/Regressed |
|---|---|---|---|---|
| dev (20 tasks, selection) | 50% | 60% | +10 pp, monotonic | +3 / −1 |
| **test (20 tasks, frozen)** | **60%** | **65%** | **+5 pp, 0 regressions** | **+1 / 0** |
| **test 3-seed pooled** | 60% | — | **mean +1.7 pp, CI [0, +5]** | **0 regressions** |

Full numbers and failure analysis: [experiment results](docs/EXPERIMENT-RESULTS.md).

## Four-Round Evolution: Every Failure Has a Root Cause

| Round | What changed | Samples | Eval set | Outcome | Root cause |
|---|---|---|---|---|---|
| 1 | Initial SFT | 71 | 2/2 tasks | SFT output byte-identical to Base | Too little data; learned style, not semantics |
| 2 | Expanded to 84 tasks | 204 | 12/12 tasks | dev +1 / test −1 | Eval set too small (noise); some hidden tests underdetermined training labels |
| 3 | Fixed labels, kept 1 solution/task | 60 | 12/12 tasks | No change | Dropped 74% of supervised tokens — data starvation |
| **4** | **104 tasks + mutation audit + 3 seeds** | **202** | **20/20 tasks** | **+1.7 pp, zero regressions** | Data ×2.8 + quality gate + eval resolution ×1.7, all three aligned |

**Key safeguard validated**: across 3 training seeds in round 4, one seed showed no dev gain — the selection mechanism automatically kept the Base model instead of deploying SFT. The "no dev gain → don't deploy" anti-regression gate holds across seeds.

## Core Methods

| Capability | Implementation |
|---|---|
| Task benchmark | 104 tasks / 52 source groups, mutation-testing data-quality audit |
| Data isolation | Source-group frozen splits (64/20/20), candidate migration by task fingerprint (56% savings) |
| SFT training | NF4 QLoRA, assistant-only mask via chat-template prefix subtraction, supervised-token-weighted gradient accumulation, per-epoch immutable checkpoints |
| Model selection | By dev **execution pass@1** (not loss), Base included, ties keep Base |
| Evaluation | Greedy default, pinned protocol hash, paired bootstrap intervals, contamination checks |
| DPO | Real trainer + FEST adaptive β + hard-negative AST clustering ([DPO experiment](docs/EXPERIMENT-DPO.md), [FEST analysis](docs/FEST-ANALYSIS.md)) |

### DPO Negative Result

DPO scored 55% on dev (below SFT's 60%), not deployed per protocol. Training log showed `preference_accuracy=1.0` throughout — teacher answers vs model failures form "already-satisfied preferences," gradients vanish. Fix (adaptive β + hard-negative) implemented per [FEST](https://arxiv.org/abs/2605.15012), awaiting GPU validation.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q           # 470 tests
python -m agentlab doctor
```

GPU training and evaluation:

```bash
python -m pip install -e ".[dev,train,api]"
python -m pip check
```

Full pipeline: [START-HERE](docs/START-HERE.md) and the [GPU post-training guide](docs/POSTTRAIN-GPU.md); `scripts/run_experiment_v2.sh` runs stage-by-stage with resume support.

## Pipeline Architecture

```text
Task pool (mutation-audited) → source-group frozen splits → train / dev / test
      ↓ train / dev only
Public instruction + buggy code + dependencies + public tests
      ↓
Teacher candidates → three-layer grading → replay verification → SFT data
      ↓
QLoRA SFT → per-epoch snapshots → dev execution selection → frozen-test eval
```

## Project Structure

```text
scripts/           Data prep, collection, training, evaluation, privacy audit
src/agentlab/
  benchmark/       Tasks, public protocol, three-layer grader, mutation audit
  training/        SFT / DPO trainers, data preparation
  runtime/         Workspaces, execution, budgets, snapshots
  trajectories/    Same-state branching and preference construction
  experiments/     Experiment contracts and run state
  providers/       OpenAI-compatible providers
docs/              Experiment results, methodology docs
tests/             Automated tests (no real model/API required)
```

## Known Limitations

- Grading executes model code; production use needs container and resource isolation.
- Full agent tool interaction and online RL not yet implemented.
- Current tasks are single-file flat-Python repairs; multi-file repair requires protocol extension.
- `data/`, `results/`, `checkpoints/` not committed.

Inspired by [pi](https://github.com/earendil-works/pi), independently implemented.

## References

- [FEST](https://arxiv.org/abs/2605.15012) — adaptive β / hard-negative selection
- [DPO](https://arxiv.org/abs/2305.18290) — Direct Preference Optimization
- [GRPO](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization

License: MIT.
