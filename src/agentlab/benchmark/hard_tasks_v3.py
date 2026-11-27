"""Hard repair tasks (batch h3): several interacting defects per program.

Unlike the default 104-task benchmark, every program here carries several
interacting defects and a fix has to keep every stated requirement true at
once.  Five parts, two variants per source family: state consistency across
mutating operations, transactions and all-or-nothing multi-step updates,
exception safety and cleanup, lock ordering with atomicity, and condition
variables with worker lifecycles.

The batch is opt-in (``EXTRA_POOLS=h3``); the default pool is unchanged.
"""
from __future__ import annotations

from agentlab.benchmark.hard_tasks_v3_cond import MUTANTS as _COND_MUTANTS
from agentlab.benchmark.hard_tasks_v3_cond import SPECS as _COND
from agentlab.benchmark.hard_tasks_v3_exc import MUTANTS as _EXC_MUTANTS
from agentlab.benchmark.hard_tasks_v3_exc import SPECS as _EXC
from agentlab.benchmark.hard_tasks_v3_lock import MUTANTS as _LOCK_MUTANTS
from agentlab.benchmark.hard_tasks_v3_lock import SPECS as _LOCK
from agentlab.benchmark.hard_tasks_v3_state import MUTANTS as _STATE_MUTANTS
from agentlab.benchmark.hard_tasks_v3_state import SPECS as _STATE
from agentlab.benchmark.hard_tasks_v3_tx import MUTANTS as _TX_MUTANTS
from agentlab.benchmark.hard_tasks_v3_tx import SPECS as _TX

HARD3_SPECS = [*_STATE, *_TX, *_EXC, *_LOCK, *_COND]
# One partial fix per stated requirement, graded by scripts/self_check_pool.py
HARD3_MUTANTS = {**_STATE_MUTANTS, **_TX_MUTANTS, **_EXC_MUTANTS, **_LOCK_MUTANTS, **_COND_MUTANTS}
