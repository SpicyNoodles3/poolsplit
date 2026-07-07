# poolsplit

[![test](https://github.com/SpicyNoodles3/poolsplit/actions/workflows/test.yml/badge.svg)](https://github.com/SpicyNoodles3/poolsplit/actions/workflows/test.yml)

Pool-split retrieval for agent memory. When one scoring function ranks a
mixed-type memory store against a single token budget, high-priority entry types
monopolize the context window and low-priority types never surface. poolsplit
gives each type-group its own reserved budget — a "pool" retrieved independently
— and lets always-scope entries (behavioral corrections) load unconditionally,
outside budget competition entirely.

## The problem

One scorer plus one budget produces starvation. High-base-priority types win
every slot; anything with low priority or a cold retrieval history never loads.
In the system this pattern came from, measured before reservation: growth
entries loaded on 0 of 4 test queries, skills on 1 of 30.

The failure has an inverse, and it is the dangerous one. A burst of new,
freshly-scored content can push a standing behavioral correction out of scope
exactly when it matters most — the moment the agent is about to repeat the
mistake the correction exists to prevent. A correction is worthless if it only
loads on quiet turns.

Longer write-up of the failure mode:
[Your agent's memory has a starvation problem](docs/starvation.md).

## The pattern

Two ideas:

- **Pools are reserved budgets.** Each pool owns a token ceiling and an entry
  cap for a group of types. A pool is filled independently, so a high-priority
  type in one pool cannot consume the budget of a low-priority type in another.
  The low-priority type gets guaranteed room.
- **Always-scope entries load unconditionally.** An entry with `scope="always"`
  — a behavioral correction — loads first and bypasses both the token ceiling
  and the entry cap. It can never be crowded out, no matter how much fresh
  content arrives.

```python
import datetime
from poolsplit import Pool, retrieve_pools

entries = [
    {"id": "c1", "type": "correction", "scope": "always",
     "content": "Never claim a write that did not happen.", "created": "2026-07-06T00:00:00Z"},
    {"id": "t1", "type": "task", "base_priority": 7,
     "content": "Ship the refactor.", "created": "2026-07-06T00:00:00Z"},
    {"id": "s1", "type": "skill", "base_priority": 2,
     "content": "grep to the symbol before reading the file.", "created": "2026-07-06T00:00:00Z"},
]
pools = [
    Pool("corrections", max_tokens=1, max_entries=8, types=("correction",)),
    Pool("content", max_tokens=200, max_entries=15, types=("task",)),
    Pool("skill", max_tokens=80, max_entries=2, types=("skill",)),
]
by_pool = retrieve_pools(entries, pools, now=datetime.datetime(2026, 7, 7))
# {"corrections": [c1], "content": [t1], "skill": [s1]}
# c1 surfaces on a 1-token budget (always-scope); s1 surfaces despite low
# priority because it has its own pool.
```

## Install

It is a single dependency-free module. Vendoring it is a supported install
method — copy `poolsplit.py` into your project and import it.

From source:

```
pip install .
```

Development (tests):

```
pip install .[dev]
pytest -q
```

Requires Python 3.9 or newer. No runtime dependencies.

## API reference

### `Pool(name, max_tokens, max_entries, types=None)`

A frozen dataclass describing one reserved budget.

- `name` — unique key under which the pool's results are returned.
- `max_tokens` — token ceiling for contextual (non-always) entries.
- `max_entries` — cap on contextual entries; always-scope entries do not count
  against it.
- `types` — tuple of entry types this pool draws from; `None` accepts all types.
- Negative `max_tokens` or `max_entries` raises `ValueError`.

### `retrieve(entries, pool, context=None, *, scorer, token_counter, is_active, now) -> List[dict]`

Fill one pool. Semantics:

- Filter to active entries, then to the pool's types.
- Split into always-scope (`scope == "always"`) and contextual entries.
- **All always-scope entries load first**, bypassing both `max_tokens` and
  `max_entries`. This is the invariant the library exists for.
- Contextual entries are scored and sorted by the fully deterministic key
  `(-score, created-or-"", id-or-"", original-index)`. Input order never
  changes the result except through the `original-index` tiebreak, which breaks
  all remaining ties, so shuffling equal-key entries cannot change the output.
- Greedy budget fill. An entry's cost is `entry["token_count"]` when present and
  a positive int, else `token_counter(content)` — **never 0 for a missing
  count**. A 0-cost default would silently exempt entries from every budget.
- Returned entries are the **same dict objects** (no copies), ordered
  always-scope (input order) then contextual (sorted order).
- Never mutates entries.

### `retrieve_pools(entries, pools, context=None, *, ...) -> Dict[str, List[dict]]`

Fill several pools from one entry set. Pools are processed in order; an entry
selected by an earlier pool is excluded from later pools, so each entry lands in
at most one pool. Dedupe is by object identity (`id()`), so entries lacking an
`"id"` key still dedupe correctly. Duplicate pool names raise `ValueError`.

### `default_scorer(entry, context, *, now=None) -> float`

Base priority × recency decay (linear over a 90-day horizon, floored at 0.3) ×
tag overlap × project match × message-content word overlap. `now` is injectable
for deterministic scoring. There is **no** access-count / frequency term (see
"What it does not do"). Lexical and fully pluggable — pass your own `scorer`.

### `default_token_counter(text) -> int`

`max(1, len(text) // 4)`. Crude by design and pluggable; never returns 0.

### `default_is_active(entry) -> bool`

True when `status` is `"active"` (the default when the key is missing) and there
is no `superseded_by` pointer.

## What it does not do

- **Persistence.** poolsplit reads a list of dicts and returns a list of dicts.
  It does not load, save, lock, or migrate a store.
- **Semantic scoring.** The default scorer is lexical — tag and word overlap, no
  embeddings. It is fully pluggable; supply your own `scorer` for vector
  similarity or anything else.
- **Access-frequency tracking.** No entry is mutated on read; there are no
  access counts and no frequency term in scoring. This is deliberate: frequency
  feedback loops (retrieving an entry raises its count, which raises its score,
  which gets it retrieved again) entrenched roughly 97% of the origin store as
  unreachable on neutral queries. Ranking is kept independent of retrieval
  history.
- **Async.** Everything is synchronous and in-process.

## Provenance

Extracted and re-expressed clean-room from the retrieval layer of a personal AI
system running in production since early 2026.

## License

MIT. See [LICENSE](LICENSE).
