# CodeAgent-R

**Post-training for code repair** · [中文](README.md)

![Python 3.10+, PyTorch NF4 QLoRA, Transformers 4.x for training, 470 tests passed, MIT license.](docs/images/project-facts.svg)

CodeAgent-R evaluates **Qwen2.5-Coder-7B-Instruct** on 104 Python repair tasks. How many more programs can it fix by learning from a teacher, preference pairs, or its own successful attempts? Each checkpoint competes with the base model (Base) on dev, then the selected model is scored task by task on frozen test data.

[Scorecard](#method-comparison) · [Three seeds](#three-training-seeds) · [Experiment history](#experiment-history) · [Pipeline](#training-and-evaluation) · [Quick start](#run-locally)

## Method comparison

![Execution pass rates on dev and test for Base, SFT, three DPO rounds, two RFT rounds, hybrid training, and Best-of-N inference.](docs/images/method-comparison.png)

**On the same 20 test tasks, Base repaired 12 programs. SFT and the first RFT round each repaired 13.** A second RFT round brought dev up from 14 to 16 repairs while test fell from 13 to 11.

| Method | Dev | Test | Setting and observation |
| --- | ---: | ---: | --- |
| Base | 10/20 (50%) | 12/20 (60%) | Base model, greedy decoding |
| **Teacher-distilled SFT** | 12/20 (60%) | **13/20 (65%)** | Seed 42 selected epoch 3 on dev |
| DPO round 1 | 11/20 (55%) | — | Frozen SFT reference, uniform β=0.1 |
| DPO round 2 | 11/20 (55%) | — | Same reference, β=0.1 or 0.01 by sampling group |
| DPO round 3 | 10/20 (50%) | — | Base reference, training loss fell from 0.72 to 0.63 |
| **RFT round 1** | 14/20 (70%) | **13/20 (65%)** | Continued from SFT on about 150 self-generated repairs |
| RFT round 2 | 16/20 (80%) | 11/20 (55%) | Continued from round 1 on 180 repairs covering 58 tasks |
| Hybrid training | — | 13/20 (65%) | Mixed 202 teacher examples with about 150 RFT examples |
| Best-of-N inference | — | 13/20 (65%) | Greedy first, then up to four extra samples screened by public tests |

All three DPO runs scored below SFT on dev and stopped there. The hybrid and Best-of-N runs report test scores only. Seven test tasks remained unsolved across the methods that reached final evaluation. [Full scorecard](docs/EXPERIMENT-RESULTS.md) · [DPO run notes](docs/EXPERIMENT-DPO.md)

## Three training seeds

![Test pass rates on the left are 65%, 60%, and 60%, with Base at 60%. The gains on the right are +5, 0, and 0 percentage points.](docs/images/three-seed-results.png)

All three training seeds share one frozen test set. Each run selects on dev first, with Base included as a candidate. Ties keep Base.

| Training seed | Dev selection | Test tasks passed | Gain over Base |
| ---: | --- | ---: | ---: |
| 42 | Epoch 3 | **13/20** | **+5 pp** |
| 43 | Base | 12/20 | 0 |
| 44 | Epoch 2 | 12/20 | 0 |

The mean gain was **1.7 percentage points**. On dev, seed 42 improved three tasks and regressed on one, for a net gain of two. On test, it fixed one additional task with zero regressions. The other two runs matched the baseline score. The 95% task-bootstrap interval was **[0, +5] percentage points**. A single repair moves the score by five points on a test set of this size. [Aggregation code](scripts/aggregate_seeds.py) · [Mobile chart](docs/images/three-seed-results-mobile.png)

## Experiment history

![The experiment trajectory shows the Base reference, SFT, DPO, and two RFT rounds. Dev and test diverge in the second RFT round.](docs/images/experiment-evolution.png)

The 50% dev and 60% test markers on the left are the baseline values used in the method comparison. The early rounds used smaller evaluation sets. Their actual sizes appear below.

| Round | Data or training change | Observation within that round |
| --- | --- | --- |
| 1 · Initial SFT | 71 examples, two tasks each on dev and test | Outputs were byte-identical to Base |
| 2 · Larger pool | 84 tasks, 204 examples, 12 tasks each on dev and test | One more dev repair and one fewer test repair |
| 3 · Label fixes | One solution per task, 60 examples | Supervised tokens fell to about 26% of the previous total and scores held steady |
| 4 · Frozen evaluation | 104 tasks, 202 teacher examples, a mutation audit and three training seeds | 20 tasks each on dev and test, mean test gain of +1.7 pp after selection |
| 5–7 · DPO | Six samples per train task across 64 tasks yielded 50 preference pairs | The three runs passed 11, 11, and 10 dev tasks. SFT passed 12 |
| 8 · RFT | About 150 self-generated examples across roughly 50 tasks | Two more dev repairs. Test matched SFT |
| 9 · Further RFT | Resampled from round 1 and reduced the learning rate from 1e-5 to 5e-6 | Two more dev repairs and two fewer test repairs than round 1 |

### Interpreting the differences

- **Examples and task coverage.** Rounds 2 and 4 had nearly identical example counts (204 and 202), yet the task pool, label audit, and evaluation set size all changed. Comparing rounds calls for independent task counts, source groups, and supervised-token counts.
- **The preference signal.** The student passed 207 of 384 samples. With an SFT reference, preference accuracy was near 1 and reward margin near zero. Establishing the size of parameter gradients would require gradient measurements. Model selection can use the observed repair count on the 20 dev tasks. [DPO analysis](docs/EXPERIMENT-DPO.md)
- **The inference budget.** Best-of-N generates extra candidates when greedy fails a public test. Its higher candidate budget produced the same 13/20 as greedy SFT. Reporting both repairs and allowed generations per task makes the cost of the comparison visible.

## Training and evaluation

![The frozen benchmark contains 104 tasks in 52 source groups. Train has 64 tasks, dev and test have 20 each. Each mark represents one source group.](docs/images/frozen-split.svg)

1. **Freeze whole source groups.** Both tasks in a group stay in the same split. The export checks each reference repair and reproduces each original bug. In the mutation audit, 75 of 104 tasks caught every mutant generated for them. [Benchmark and grader](src/agentlab/benchmark/) · [Mutation audit](scripts/audit_mutations.py)
2. **Replay teacher candidates.** The teacher receives the instruction, buggy code, dependencies, and public tests. Data preparation deduplicates and replays passing repairs. The reported SFT run retained 202 examples from 64 training tasks. Reusing candidates with matching fingerprints reduced the reported collection cost by 56%. [Data preparation](scripts/prepare_data.py) · [Candidate migration](scripts/migrate_candidates.py)
3. **Train and keep each epoch.** SFT used double-quantized NF4 QLoRA, rank 16, α=32, and a learning rate of 2e-5. Only target assistant tokens contributed to the loss. Gradient accumulation was weighted by supervised-token count, with 78 optimizer steps over three epochs. [SFT implementation](src/agentlab/training/hf_sft.py)
4. **Select on dev, then evaluate on test.** Compare Base and every epoch on dev execution pass@1 with the same decoding budget. Once selection is fixed, count gains and regressions on matched test tasks. [Checkpoint selection](scripts/select_checkpoint.py) · [Paired comparison](scripts/compare.py)

Two follow-up training paths start from supervised fine-tuning (SFT). **Direct preference optimization (DPO)** pairs a passing teacher solution with a student failure under the same prompt. The adaptive variant chooses β by sampling group and uses syntax-tree fingerprints to select common failures. **Rejection sampling fine-tuning (RFT)** samples eight times per task and keeps verified repairs. A diagnostic hint can help recover a failed attempt, and the resulting training example uses the original prompt. [Preference generation](scripts/generate_preferences.py) · [RFT data generation](scripts/generate_rft_data.py)

## Run locally

From the repository root, use Python 3.10 or later. This CPU example prints the model-visible task prompt. The final `canary 在 prompt 中: False` confirms that the private marker was removed.

```bash
python -m pip install -e .
python -m agentlab doctor
python examples/p01_minimal.py
```

Run the full offline suite on Linux or WSL2.

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

GPU training and evaluation use the `.[dev,train,api]` extra. To reproduce the 64/20/20 split, set `--dev-ratio 0.20 --test-ratio 0.20` explicitly. Teacher collection sends paid requests. Use `--status-only` to inspect candidate slots first. Both guides are in Chinese. [GPU guide](docs/POSTTRAIN-GPU.md) · [Getting started](docs/START-HERE.md)

## Repository and scope

| Area | Code and documentation |
| --- | --- |
| Task specifications, public inputs, and patch grading | [benchmark](src/agentlab/benchmark/) · [contracts](src/agentlab/contracts/) |
| SFT and DPO training, experiment stages and state | [training](src/agentlab/training/) · [experiments](src/agentlab/experiments/) · [scripts](scripts/) |
| Workspaces, tool budgets, trajectories, and providers | [runtime](src/agentlab/runtime/) · [trajectories](src/agentlab/trajectories/) · [providers](src/agentlab/providers/) |
| Detailed runs, preference optimization, and regression tests | [Run records](docs/EXPERIMENT-RESULTS.md) · [FEST implementation notes](docs/FEST-ANALYSIS.md) · [tests](tests/) |

Each task replaces a single Python file. Dependencies are read-only context. The grader executes candidate code, so untrusted candidates need container isolation and resource limits. Full tool-use episodes and online RL remain future work.

Public reports share aggregate metrics and public model revisions. Teacher credentials, private tests, per-task outputs, raw logs, and checkpoints stay local. [Sharing results](docs/START-HERE.md#分享实验结果)

## Related work

- **Code repair and execution feedback** · [SWE-Gym](https://arxiv.org/abs/2412.21139) · [FailForge](https://arxiv.org/abs/2608.08570) · [PerfCodeGen](https://arxiv.org/abs/2412.03578) · [Self-Execution Simulation](https://arxiv.org/abs/2604.03253)
- **Self-generated data and rejection sampling** · [GiFT](https://arxiv.org/abs/2502.11466) · [RFT](https://arxiv.org/abs/2308.01825)
- **Preferences and group feedback** · [DPO](https://arxiv.org/abs/2305.18290) · [FEST](https://arxiv.org/abs/2605.15012) · [GRPO](https://arxiv.org/abs/2402.03300)

CodeAgent-R is an independent implementation. Its runtime design draws on pi. [pi](https://github.com/earendil-works/pi) · [MIT license](LICENSE)
