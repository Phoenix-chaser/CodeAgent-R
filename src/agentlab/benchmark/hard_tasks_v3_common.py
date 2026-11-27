"""Shared builder for the hard batch (h3).

Every task is one main file with several interacting defects, optionally
reading read-only dependency files, an instruction that states every
requirement, a public test that only shows the interface, and private tests
that cover each requirement on its own.
"""
from __future__ import annotations

from textwrap import dedent


def _task(name, group, instruction, buggy, fixed, public_tests, private_tests,
          extra_files=None, difficulty="L3"):
    """Build one h3 task spec.

    ``private_tests`` is either one code block (written to ``test_regression.py``)
    or a mapping ``{label: code}`` (each written to ``test_<label>.py``).  Every
    block gets ``import pytest`` and ``from <module> import *`` prepended.
    ``extra_files`` maps a module name (without ``.py``) to its read-only source.
    """
    module = f"h3_{name}"
    if isinstance(private_tests, str):
        private_tests = {"regression": private_tests}
    tests = {f"test_{label}.py": f"import pytest\nfrom {module} import *\n\n"
             + dedent(body).strip() + "\n"
             for label, body in private_tests.items()}
    spec = {
        "task_id": f"h3_{name}_001", "source_group": f"h3_{group}",
        "split": "train", "bug_family": group, "difficulty": difficulty,
        "instruction": instruction, "module": f"{module}.py",
        "buggy": dedent(buggy).strip() + "\n",
        "fixed": dedent(fixed).strip() + "\n",
        "public_test": f"from {module} import *\n\n" + dedent(public_tests).strip() + "\n",
        "private_tests": tests,
    }
    if extra_files:
        spec["extra_files"] = {f"{filename}.py": dedent(source).strip() + "\n"
                               for filename, source in extra_files.items()}
    # 划分时把这一批作为单独的一层按比例分给 train / dev / test（preparation.freeze_splits）
    spec["stratum"] = "hard"
    return spec
