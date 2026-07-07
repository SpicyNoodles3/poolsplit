# Your agent's memory has a starvation problem

*July 2026*

An agent with persistent memory stores more than one kind of thing. Facts. Tasks.
Procedures. Skills. And corrections — the entries written when the user pushed back:
*never claim a write that didn't happen. Stop re-deriving the system from stale state.*

Most memory systems retrieve from this store the same way: one scoring function ranks
every entry, and the winners are injected into context until a token budget runs out.
One scorer, one budget.

That design starves whole categories. Permanently.

Here are numbers from the system this pattern was extracted from, measured before the
fix. Growth-type entries loaded on 0 of 4 test queries. Skills loaded on 1 of 30. Not
ranked low and occasionally missing — structurally absent. The system had learned
things it could never remember.

## Why tuning doesn't fix it

The failure is structural. Memory types differ in base priority and recency profile,
so a global ranking is an auction in which the same types win every slot. You can
rebalance the weights, and the bottom moves — but a single ranking always has a
bottom, and whatever lives there is invisible to the agent.

It gets worse if the scorer rewards access frequency, which seems like an obvious
thing to do — surface what keeps being useful. But retrieval raises the count, the
count raises the score, the score drives retrieval. In the origin system, that loop
had entrenched roughly 97% of the store as unreachable on a neutral query. The store
was large and alive; the working set was tiny and frozen.

## The inverse failure is the one that costs you

Starvation has an inverse. A burst of fresh, high-scoring content — a busy session, a
new project, a pasted document — can push a standing behavioral correction out of
scope. So the correction fails to load on precisely the kind of turn it exists for:
the loaded, messy one where the agent is about to repeat the mistake.

A correction that only surfaces on quiet turns is decorative. If you have ever told an
agent the same thing three times, this is a likely mechanism. The correction is in the
store. It scored below a transcript chunk.

## The fix is a partition, not a better scorer

Two mechanisms.

**Reserved budgets per type.** Each type-group gets its own token ceiling and entry
cap — a pool — and each pool fills independently. A hot type in one pool cannot spend
another pool's budget. The low-priority type stops competing against the high-priority
type at all; it only competes with itself.

**An unconditional tier for corrections.** An entry marked always-scope loads on every
turn, before the scored pass, outside both the token ceiling and the entry cap. It
cannot be crowded out, no matter what arrived this week. This is the invariant that
matters: behavioral corrections are not content to be ranked. They are constraints the
ranking must not be able to displace.

Both mechanisms are boring. That is the point. Fairness across memory types is a
structural property — you cannot get it out of scoring weights, for the same reason
you cannot get a guaranteed minimum out of an auction.

## What the popular tools do instead

Before writing this I read the source of two popular open-source projects in this
space, to check whether the problem was already solved. It isn't, and the pattern of
what's missing is consistent.

One is a memory layer with over ten thousand GitHub stars. Its retrieval stack is
genuinely ahead of anything I have built — hybrid sparse-plus-dense search, calibrated
score fusion, fact-level embeddings, rerankers. But retrieval is top-k counts with
similarity floors: there is no token budget anywhere in the retrieval path, no
reserved capacity per type, and no pinned or always-load tier — every entry competes
on relevance, every time. Its confidence scores are emitted once by an LLM at
extraction and never move afterward. A grep across its memory subsystem for any
correction, feedback, or pushback mechanism returns nothing; a user saying "that's
wrong" has no write path. Supersession exists only as deduplication — an entry is
retired when it is similar enough to another to be merged, never because it was
contradicted.

The other project makes the opposite bet: it injects every learned lesson into the
prompt wholesale, every turn — no retrieval gate at all — and fights bloat at write
time with LLM-driven curation. (It also keeps a per-lesson effectiveness ledger,
helpful and harmful counts cited by ID, which is a genuinely good idea.) Wholesale
injection is coherent at the scale it targets, around fifteen lessons. It cannot
survive a store with thousands of mixed-type entries. At that point you need
retrieval; the moment you have retrieval you have ranking; the moment you have ranking
you have a bottom.

So the two poles are occupied — rank everything on one axis, or inject everything and
rank nothing. The middle position, reserved budgets per type with an unconditional
tier for corrections, is mostly empty. That surprised me, because it is the position
you get forced into by running one of these systems long enough for the user to push
back on it.

## The library

I extracted the partition logic into
[poolsplit](https://github.com/SpicyNoodles3/poolsplit) — a single dependency-free
Python module, MIT. It does the two things above and deliberately nothing else: no
persistence, no embeddings, no access-frequency tracking (the 97% loop is why), a
pluggable scorer, deterministic output. Vendoring the file is a supported install
method.

It is extracted clean-room from the retrieval layer of a personal AI system that has
been in production since early 2026. The numbers in this post are that system's.

If your agent has ever needed to be told something three times, look at what's at the
bottom of its ranking.
