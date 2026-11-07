# -*- coding: utf-8 -*-
"""v2c scale-up task pack (batch 3): schema, dedup and audit-shape regression."""
import ast

import pytest

from agentlab.benchmark.builder import SPECS
from agentlab.benchmark.pilot_tasks import PILOT_SPECS
from agentlab.benchmark.pilot_tasks_v2 import V2_SPECS
from agentlab.benchmark.pilot_tasks_v2b import V2B_SPECS
from agentlab.benchmark.pilot_tasks_v2c import V2C_SPECS
from agentlab.benchmark.protocol import validate_specs


def test_v2c_schema_valid():
    validate_specs(V2C_SPECS)
    assert len(V2C_SPECS) == 20
    assert len({spec["source_group"] for spec in V2C_SPECS}) == 10
    from collections import Counter
    counts = Counter(spec["source_group"] for spec in V2C_SPECS)
    assert set(counts.values()) == {2}


def test_v2c_disjoint_from_existing_pools():
    existing = SPECS + PILOT_SPECS + V2_SPECS + V2B_SPECS
    ids = {spec["task_id"] for spec in existing}
    groups = {spec["source_group"] for spec in existing}
    modules = {spec["module"] for spec in existing}
    asts = {ast.dump(ast.parse(spec["buggy"])) for spec in existing}
    for spec in V2C_SPECS:
        assert spec["task_id"] not in ids
        assert spec["source_group"] not in groups
        assert spec["module"] not in modules
        assert ast.dump(ast.parse(spec["buggy"])) not in asts


@pytest.mark.parametrize("spec", V2C_SPECS, ids=lambda spec: spec["task_id"])
def test_v2c_reference_and_tests_are_valid_python(spec):
    ast.parse(spec["fixed"])
    ast.parse(spec["public_test"])
    for content in spec["private_tests"].values():
        ast.parse(content)


def test_v2c_difficulty_mix():
    from collections import Counter
    mix = Counter(spec["difficulty"] for spec in V2C_SPECS)
    assert mix["L2"] == 18
    assert mix["L3"] == 2


def test_template_reference_rejects_nested_key_but_preserves_escaped_braces():
    spec = next(spec for spec in V2C_SPECS if spec["task_id"] == "v2c_render_template_001")
    namespace = {}
    exec(compile(spec["fixed"], spec["module"], "exec"), namespace)
    render = namespace["render_template"]
    with pytest.raises(ValueError):
        render('{a{b}', {'a{b': 'invalid'})
    assert render('{{{key}}}', {'key': 'ok'}) == '{ok}'
    assert render('{a}', {'a': '{b}'}) == '{b}'
