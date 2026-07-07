"""Runnable demo: reserved pools rescue low-priority and always-scope entries.

Run: python3 example.py

Shows (a) an always-scope correction surfacing despite a one-token budget, and
(b) a low-priority skill surfacing because it has its own reserved pool instead
of competing in the shared content pool.
"""

import datetime

from poolsplit import Pool, retrieve_pools

NOW = datetime.datetime(2026, 7, 7, tzinfo=datetime.timezone.utc)  # fixed for reproducibility
CREATED = "2026-07-06T00:00:00Z"

entries = [
    {"id": "c1", "type": "correction", "scope": "always", "base_priority": 6,
     "content": "Never claim a filesystem write that did not happen.", "created": CREATED},
    {"id": "c2", "type": "correction", "base_priority": 6,
     "content": "Prefer editing an existing file over creating a new one.", "created": CREATED},
    {"id": "t1", "type": "task", "base_priority": 7,
     "content": "Ship the retrieval refactor before Friday.", "created": CREATED},
    {"id": "t2", "type": "task", "base_priority": 7,
     "content": "Rotate the exposed API key.", "created": CREATED},
    {"id": "f1", "type": "declarative", "base_priority": 5,
     "content": "The service reads config from an environment variable.", "created": CREATED},
    {"id": "f2", "type": "declarative", "base_priority": 5,
     "content": "Timestamps are stored in UTC.", "created": CREATED},
    {"id": "f3", "type": "declarative", "base_priority": 5,
     "content": "The build output goes to the dist directory.", "created": CREATED},
    {"id": "s1", "type": "skill", "base_priority": 2,
     "content": "Use grep to jump to a symbol before reading a whole file.", "created": CREATED},
]

pools = [
    Pool("corrections", max_tokens=1, max_entries=8, types=("correction",)),  # tiny budget on purpose
    Pool("content", max_tokens=200, max_entries=15, types=("task", "declarative")),
    Pool("skill", max_tokens=80, max_entries=2, types=("skill",)),
]

by_pool = retrieve_pools(entries, pools, context={"tags": [], "project": None}, now=NOW)
for pool_name, selected in by_pool.items():
    ids = ", ".join(e["id"] for e in selected) or "(none)"
    print("%-12s -> %s" % (pool_name, ids))
