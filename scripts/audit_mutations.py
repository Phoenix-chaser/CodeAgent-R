"""Mutation-based test adequacy audit for task pools.

A surviving mutant = a program that behaves differently from the reference
solution yet still passes the public+private gate.  Teacher answers equal to
such a mutant would be accepted as training labels, so survivor count per task
quantifies exactly the "hidden tests underdetermine behavior" defect.

Tests run in-process with a minimal pytest stub: no subprocess, no framework
load, near-zero extra memory.  A SIGALRM guard stops mutant-induced infinite
loops.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import importlib.util
import json
import signal
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.benchmark.protocol import load_specs


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


class _Raises:
    def __init__(self, expected):
        self.expected = expected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError("DID NOT RAISE")
        return issubclass(exc_type, self.expected)


def _pytest_stub() -> types.ModuleType:
    stub = types.ModuleType("pytest")
    stub.raises = _Raises

    def fail(message=""):
        raise AssertionError(message)

    stub.fail = fail
    return stub

COMPARE_SWAPS = {
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.Lt: ast.Gt, ast.Gt: ast.Lt,
    ast.LtE: ast.GtE, ast.GtE: ast.LtE,
    ast.Lt: ast.LtE, ast.Gt: ast.GtE,
}
BINOP_SWAPS = {ast.Add: ast.Sub, ast.Sub: ast.Add}
CONST_LIMIT = 10_000


def _collect_sites(tree):
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = type(node.ops[0])
            if op in (ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE):
                swap = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.Gt,
                        ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}[op]
                sites.append(("cmp_swap", node, ("ops", swap)))
                if op in (ast.Lt, ast.Gt, ast.LtE, ast.GtE):
                    tighten = {ast.Lt: ast.LtE, ast.Gt: ast.GtE,
                               ast.LtE: ast.Lt, ast.GtE: ast.Gt}[op]
                    sites.append(("cmp_boundary", node, ("ops", tighten)))
        elif isinstance(node, ast.BoolOp):
            swap = ast.Or if isinstance(node.op, ast.And) else ast.And
            sites.append(("bool_swap", node, ("op", swap)))
        elif isinstance(node, ast.BinOp) and type(node.op) in BINOP_SWAPS:
            sites.append(("arith_swap", node, ("op", BINOP_SWAPS[type(node.op)])))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                sites.append(("bool_flip", node, ("value", not node.value)))
            elif isinstance(node.value, int) and abs(node.value) <= CONST_LIMIT:
                sites.append(("const_plus1", node, ("value", node.value + 1)))
                sites.append(("const_minus1", node, ("value", node.value - 1)))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            sites.append(("not_remove", node, ("unwrap", None)))
    return sites


def build_mutants(source: str, cap: int):
    tree = ast.parse(source)
    sites = _collect_sites(tree)
    by_kind = {}
    for site in sites:
        by_kind.setdefault(site[0], []).append(site)
    order = []
    kinds = sorted(by_kind)
    index = 0
    while len(order) < len(sites):
        progressed = False
        for kind in kinds:
            if index < len(by_kind[kind]):
                order.append(by_kind[kind][index])
                progressed = True
                if len(order) >= cap:
                    break
        if not progressed:
            break
        index += 1
    reference_source = ast.unparse(tree)

    class SingleMutator(ast.NodeTransformer):
        def __init__(self, target_dump, action):
            self.target_dump = target_dump
            self.action = action
            self.done = False

        def visit(self, node):
            if not self.done and ast.dump(node) == self.target_dump:
                self.done = True
                return self._apply(node)
            return self.generic_visit(node)

        def _apply(self, node):
            field, value = self.action
            if field == "unwrap":
                return node.operand
            if field == "value":
                node.value = value
                return node
            setattr(node, field, value())
            return node

    mutants = []
    for kind, node, action in order[:cap]:
        try:
            mutator = SingleMutator(ast.dump(node), action)
            mutated = mutator.visit(copy.deepcopy(tree))
            ast.fix_missing_locations(mutated)
            text = ast.unparse(mutated)
            if mutator.done and text != reference_source:
                mutants.append((kind, text))
        except Exception:
            continue
    return mutants


def run_tests(module_name: str, module_source: str, public_test: str,
              private_tests: dict, workdir: Path, timeout_s: float = 5.0) -> bool:
    """Run public+private test functions in-process against module_source."""
    workdir.mkdir(parents=True, exist_ok=True)
    source_path = workdir / f"subject_{time.monotonic_ns()}.py"
    source_path.write_text(module_source, encoding="utf-8")
    real_pytest = sys.modules.get("pytest")
    had_module = module_name in sys.modules
    old_module = sys.modules.get(module_name)
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules["pytest"] = _pytest_stub()
    try:
        module_spec = importlib.util.spec_from_file_location(module_name, source_path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        namespace = {"__name__": "mutation_audit_tests"}
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            for content in [public_test, *private_tests.values()]:
                text = content if "import pytest" in content else f"import pytest\n{content}"
                exec(compile(text, "<test>", "exec"), namespace)
            for name in sorted(namespace):
                if name.startswith("test_") and callable(namespace[name]):
                    namespace[name]()
        except _Timeout:
            return False
        except BaseException:
            return False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return True
    finally:
        sys.modules.pop("pytest", None)
        if real_pytest is not None:
            sys.modules["pytest"] = real_pytest
        sys.modules.pop(module_name, None)
        if had_module and old_module is not None:
            sys.modules[module_name] = old_module
        sys.dont_write_bytecode = old_dont_write
        with contextlib.suppress(OSError):
            source_path.unlink()


def audit_task(spec: dict, cap: int) -> dict:
    module = spec["module"][:-3]
    with tempfile.TemporaryDirectory(prefix="mutaudit-") as tmp:
        work = Path(tmp)
        if not run_tests(module, spec["fixed"], spec["public_test"],
                         spec["private_tests"], work):
            return {"task_id": spec["task_id"], "status": "reference_fails",
                    "mutants": 0, "survivors": 0, "survivor_kinds": []}
        mutants = build_mutants(spec["fixed"], cap)
        survivors = []
        for kind, source in mutants:
            if run_tests(module, source, spec["public_test"],
                         spec["private_tests"], work / "m"):
                survivors.append(kind)
        return {"task_id": spec["task_id"], "source_group": spec["source_group"],
                "status": "ok", "mutants": len(mutants),
                "survivors": len(survivors), "survivor_kinds": survivors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--cap", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    specs = load_specs(args.tasks_file)
    rows = []
    started = time.time()
    for position, spec in enumerate(specs, 1):
        row = audit_task(spec, args.cap)
        rows.append(row)
        detail = f" ({','.join(row['survivor_kinds'])})" if row["survivor_kinds"] else ""
        print(f"[{position}/{len(specs)}] {row['task_id']}: "
              f"{row['survivors']}/{row['mutants']} survivors{detail}", flush=True)
    clean = [r for r in rows if r["status"] == "ok"]
    report = {
        "schema_version": 1,
        "tasks_file": args.tasks_file,
        "tasks_sha256_entries": len(specs),
        "cap_per_task": args.cap,
        "elapsed_s": round(time.time() - started, 1),
        "quarantine_candidates": [r for r in clean if r["survivors"] >= 3],
        "summary": {
            "audited": len(clean),
            "reference_failures": len(rows) - len(clean),
            "zero_survivors": sum(1 for r in clean if r["survivors"] == 0),
            "one_or_two": sum(1 for r in clean if 0 < r["survivors"] < 3),
            "three_or_more": sum(1 for r in clean if r["survivors"] >= 3),
        },
        "tasks": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
