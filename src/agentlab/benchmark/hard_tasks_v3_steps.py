"""Single-defect steps towards the hard batch (h3).

Every parent task of the hard batch carries several interacting defects.  A
step is the same program, the same instruction and the same tests with exactly
one stated requirement broken: the reference solution with one of the partial
fixes from ``HARD3_MUTANTS`` applied.  The model has to find that one defect
and repair it without breaking anything else, which is a rung below the parent
task, where the base model scores zero.

A step shares the source group of its parent, so it always lands in the same
split.  Only mutants that pass the public test and are failed by the hidden
tests are used (183 of 197); the others break or hang the
public test.  Two mutants that lead to the same program count once.  The pool export and
``scripts/self_check_pool.py`` replay every step.
"""
from __future__ import annotations

from agentlab.benchmark.hard_tasks_v3 import HARD3_MUTANTS, HARD3_SPECS

# Indices into HARD3_MUTANTS[task_id]: the mutants whose public test still passes.
# Graded on 2026-11-24 against the hidden tests with a 30 second limit.
PUBLIC_PASSES = {
    "h3_atomic_write_001": (0, 1, 2, 3, 4),
    "h3_bounded_channel_001": (0, 1, 2, 3),
    "h3_bounded_channel_drain_001": (0, 1, 2, 4),
    "h3_cleanup_scope_001": (0, 1, 2, 3, 4),
    "h3_clock_registry_001": (0, 1, 2, 3, 4),
    "h3_conn_session_001": (0, 1, 2, 3, 4),
    "h3_delay_scheduler_001": (0, 1, 2, 3),
    "h3_delay_scheduler_wake_001": (0, 1, 2, 3, 4),
    "h3_key_index_table_001": (0, 1, 2, 3, 4),
    "h3_ledger_post_001": (0, 1, 2, 3, 4),
    "h3_nested_tx_001": (0, 1, 2),
    "h3_norm_key_index_001": (1, 2, 3),
    "h3_pair_bimap_001": (0, 1, 2, 3, 4),
    "h3_pipeline_stage_001": (0, 1, 2, 4),
    "h3_pipeline_stage_stop_001": (0, 1, 2, 3, 4),
    "h3_policy_bimap_001": (0, 1, 2, 3, 4),
    "h3_priced_bill_001": (0, 1, 2, 3, 4),
    "h3_recover_batch_001": (0, 1, 2, 3, 4),
    "h3_replan_graph_001": (0, 1, 2, 3, 4),
    "h3_resource_pool_001": (0, 1, 2, 3, 4),
    "h3_restat_graph_001": (0, 1, 2, 3, 4),
    "h3_retry_batch_001": (0, 1, 2, 3, 4),
    "h3_retry_step_001": (0, 1, 2, 4),
    "h3_rw_lock_prefer_001": (0, 1, 3),
    "h3_rw_lock_upgrade_001": (0, 1, 3),
    "h3_session_lru_001": (0, 1, 2, 3, 4),
    "h3_shard_counter_bulk_001": (0, 1, 2, 3, 4),
    "h3_shard_counter_snapshot_001": (0, 1, 2, 3, 4),
    "h3_single_flight_cache_001": (0, 1, 2, 3),
    "h3_single_flight_ttl_001": (0, 1, 2, 3, 4),
    "h3_staged_save_001": (1, 2, 3, 4),
    "h3_stock_totals_001": (0, 1, 2, 3, 4),
    "h3_teller_transfer_001": (0, 1, 2, 3, 4),
    "h3_transfer_lock_bank_001": (0, 1, 2, 3, 4),
    "h3_transfer_lock_limit_001": (0, 1, 2, 3, 4),
    "h3_undo_batch_001": (0, 1, 2, 3, 4),
    "h3_work_abort_001": (0, 1, 2, 3, 4),
    "h3_work_commit_001": (0, 1, 2, 3, 4),
    "h3_worker_pool_001": (0, 1, 2, 3),
    "h3_worker_pool_take_001": (0, 1, 3, 4),
}


def _step(parent: dict, index: int) -> dict:
    old, new = HARD3_MUTANTS[parent["task_id"]][index]
    return {
        **parent,
        "task_id": f"h3v_{parent['task_id'][3:-4]}_{index + 1}",
        "buggy": parent["fixed"].replace(old, new),
        "variant_of": parent["task_id"],
    }


STEP_SPECS = [_step(parent, index) for parent in HARD3_SPECS
              for index in PUBLIC_PASSES.get(parent["task_id"], ())]
