# CodeAgent-R

[中文](README.md) · **English**

Give a model a buggy Python file, ask for a repair, then run the tests to see how many more tasks it can solve. CodeAgent-R uses this protocol to compare supervised fine-tuning (SFT), direct preference optimization (DPO), rejection sampling fine-tuning (RFT), and inference-time sampling on Qwen2.5-Coder-7B-Instruct.

[Run locally](#run-locally) · [Results](#results-on-the-frozen-test-set) · [GPU guide (in Chinese)](docs/POSTTRAIN-GPU.md) · [MIT license](LICENSE)

## Results on the frozen test set

<picture>
  <source media="(max-width: 600px)" srcset="docs/images/three-seed-results-mobile.png">
  <img alt="Base passed 12 of 20 tasks. After checkpoint selection on dev, three training seeds passed 13, 12, and 12 tasks on the same test set. The mean gain over Base was 1.7 percentage points." src="docs/images/three-seed-results.png">
</picture>

On this set, **one more passing repair is worth 5 percentage points**. Seed 42 selected epoch 3 and solved one extra task. Seed 43 kept Base on dev. Seed 44 selected epoch 2 and matched Base on test. The 95% task-bootstrap interval for the mean gain was **0 to +5 percentage points**.

The first RFT round passed 14/20 on dev and 13/20 on test. A second round reached 16/20 on dev while test fell to 11/20. All three DPO variants scored below SFT on dev and stopped there under the selection protocol. [Full comparisons and run settings](docs/EXPERIMENT-RESULTS.md)

## Run locally

From the repository root, use Python 3.10 or later. Once the base package is installed, the example runs on CPU and requires no teacher API key.

```bash
python -m pip install -e .
python -m agentlab doctor
python examples/p01_minimal.py
```

The example prints the task prompt that the model sees. Its last line includes `canary 在 prompt 中: False` when the private marker has been stripped. For the full test suite, run `python -m pip install -e ".[dev]"` followed by `python -m pytest tests -q`. [Getting-started guide (in Chinese)](docs/START-HERE.md)

## How a repair is selected

| Stage | Protocol |
| --- | --- |
| Freeze the tasks | The 104 single-file repairs come from 52 source groups. Each group stays in one split. The reported run used 64 train, 20 dev, and 20 test tasks. |
| Verify candidates | The model sees the instruction, buggy code, dependencies, and public tests. The grader runs hidden tests separately. Data preparation replays candidate repairs. |
| Train and select | NF4 QLoRA trains on verified answers, supervising only the target assistant tokens. Each epoch produces an adapter. Base competes on the same dev execution metric and wins ties. |
| Evaluate | With the model revision and decoding budget fixed, compare the selected model with Base on each frozen test task. Count both gains and regressions. |

## Explore the repository

| Question | Where to look |
| --- | --- |
| How are tasks built, graded, and isolated across splits? | [Benchmark and grader](src/agentlab/benchmark/) · [Mutation audit](scripts/audit_mutations.py) |
| How do training, selection, and evaluation fit together? | [Training code](src/agentlab/training/) · [Experiment scripts](scripts/) · [GPU guide](docs/POSTTRAIN-GPU.md) |
| Why did the preference experiments stop on dev? | [Three DPO rounds](docs/EXPERIMENT-DPO.md) · [FEST implementation notes](docs/FEST-ANALYSIS.md) |
| How are workspaces, budgets, and trajectories recorded? | [Runtime](src/agentlab/runtime/) · [Same-state branching](src/agentlab/trajectories/) · [Experiment contracts](contracts/) |

The repair protocol replaces one Python file. Dependencies are read-only context. Grading executes candidate code, so untrusted candidates require a separate container and resource limits. Full tool-use episodes and online RL remain future work.

Keep teacher credentials, raw generations, per-task logs, and checkpoints local. Public reports can share aggregate scores and model revisions. [Sharing results (中文)](docs/START-HERE.md#分享实验结果)

CodeAgent-R is an independent implementation. Its runtime design draws on [pi](https://github.com/earendil-works/pi).
