"""Cross-seed aggregation must resample source groups, not individual tasks."""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_seeds import aggregate, load_pair  # noqa: E402
from evaluate import content_hash  # noqa: E402

GROUPS = {"a1": "a", "a2": "a", "b1": "b", "b2": "b",
          "c1": "c", "c2": "c", "d1": "d", "d2": "d"}
UP = ["a1", "a2", "c1", "c2"]


def report(passing, with_groups=True):
    generation = {"repeats": 1, "seed": 42}
    protocol = {"generation": generation, "split": "test", "benchmark_hash": "tasks",
                "prompt_hashes": {task: f"prompt-{task}" for task in GROUPS}}
    results = [dict(task_id=task, repeat_index=0, strict_pass=task in passing, seed=42,
                    prompt_hash=f"prompt-{task}",
                    verdict="strict_pass" if task in passing else "strict_fail")
               for task in sorted(GROUPS)]
    if with_groups:
        for record in results:
            record["source_group"] = GROUPS[record["task_id"]]
    return dict(schema_version="codeagent-eval-v2", model="model", resolved_revision="sha",
                adapter=None, split="test", tasks_hash="tasks", protocol=protocol,
                protocol_hash=content_hash(protocol), task_ids=sorted(GROUPS),
                task_hashes={task: f"{task}-hash" for task in GROUPS},
                contamination={"status": "clean"}, generation=generation, results=results)


def pair(label, with_groups=True):
    """One seed: the fix flips group a and c up, group b and d down."""
    base = report(set(GROUPS) - set(UP), with_groups)
    sft = report(UP, with_groups)
    with tempfile.TemporaryDirectory() as temporary:
        paths = []
        for name, data in (("base", base), ("sft", sft)):
            path = Path(temporary) / f"{name}.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            paths.append(str(path))
        return load_pair(label, *paths)


def test_seed_interval_clusters_by_source_group():
    pairs = [pair("42"), pair("43")]
    report_ = aggregate(pairs, samples=4000, seed=42)
    assert report_["cluster_by"] == "source_group" and report_["source_groups"] == 4
    assert report_["mean_delta_ci_pp"] == [-100.0, 100.0]
    historical = aggregate(pairs, samples=4000, seed=42, cluster_by="task")
    assert historical["mean_delta_ci_pp"] == [-75.0, 75.0]   # 按题重采样更窄
    assert historical["source_groups"] is None


def test_seed_interval_needs_groups_on_every_task():
    pairs = [pair("42", with_groups=False)]
    with pytest.raises(ValueError, match="source_group"):
        aggregate(pairs, samples=100, seed=42)
    assert aggregate(pairs, samples=100, seed=42, cluster_by="task")["cluster_by"] == "task"
    with pytest.raises(ValueError):
        aggregate(pairs, samples=100, seed=42, cluster_by="repository")
