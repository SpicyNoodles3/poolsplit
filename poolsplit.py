"""poolsplit — pool-split retrieval for agent memory.

When a single scoring function ranks a mixed-type memory store against a fixed
token budget, high-base-priority entry types monopolize the context window and
low-priority types never surface. poolsplit fixes this by giving each type-group
its own *reserved* token budget (a "pool") retrieved independently, and by
letting always-scope entries (behavioral corrections) load outside budget
competition entirely.

Entries are plain dicts, so the library is zero-friction with any JSON-backed
store. Recognized keys, all optional except ``content``:

    content        str            the payload text
    type           str            entry type; default "note"
    scope          str            "always" loads unconditionally; anything else
                                  competes for a pool's budget
    tags           list[str]      topical tags
    project        str | None     owning project, or None for "general"
    created        str            ISO-8601 timestamp; drives recency decay
    token_count    int            precomputed cost; else computed from content
    base_priority  float          ranking weight; default 5
    status         str            "active" (default) loads; anything else drops
    superseded_by  str | None     when set, the entry is retired and drops
    id             str            stable identifier (used in the sort tiebreak)

There are no runtime dependencies. Python >= 3.9.
"""

from dataclasses import dataclass
import datetime
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Pool",
    "default_token_counter",
    "default_is_active",
    "default_scorer",
    "retrieve",
    "retrieve_pools",
]

__version__ = "0.1.0"


# ── Pool ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pool:
    """A reserved retrieval budget for a group of entry types.

    A pool caps how much a group of types may contribute to the assembled
    context, independent of every other pool. This is the reservation that
    stops a high-priority type from consuming the whole budget.

    Attributes:
        name: Identifier for the pool; the key under which its results are
            returned by :func:`retrieve_pools`. Must be unique across the pool
            list passed to :func:`retrieve_pools`.
        max_tokens: Token ceiling for contextual (non-always) entries. Always-
            scope entries are exempt (see :func:`retrieve`). 0 admits only
            always-scope entries.
        max_entries: Cap on the number of contextual entries. Always-scope
            entries do not consume these slots. 0 admits only always-scope
            entries.
        types: Tuple of entry types this pool draws from. ``None`` means the
            pool accepts every type.

    Raises:
        ValueError: if ``max_tokens`` or ``max_entries`` is negative.
    """

    name: str
    max_tokens: int
    max_entries: int
    types: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.max_tokens < 0:
            raise ValueError("Pool.max_tokens must be >= 0, got %r" % (self.max_tokens,))
        if self.max_entries < 0:
            raise ValueError("Pool.max_entries must be >= 0, got %r" % (self.max_entries,))


# ── Defaults: token counting, activeness, scoring ───────────────────────────


def default_token_counter(text: str) -> int:
    """Estimate the token cost of a string.

    Crude by design: ``max(1, len(text) // 4)`` approximates the common
    "~4 characters per token" rule for English text. Never returns 0, so an
    entry always has a real cost against a pool's budget. Pass your own
    ``token_counter`` to :func:`retrieve` / :func:`retrieve_pools` to plug in a
    real tokenizer.

    Args:
        text: The string to measure.

    Returns:
        An estimated token count, always >= 1.
    """
    return max(1, len(text) // 4)


def default_is_active(entry: Dict) -> bool:
    """Report whether an entry is live and should be considered for retrieval.

    An entry is active when its ``status`` is "active" *and* it carries no
    ``superseded_by`` pointer.

    Deliberate improvement over the production origin: a **missing** ``status``
    counts as active. The origin required ``status == "active"`` literally,
    which silently excluded every entry that never set the field. Here the
    default is active, so an entry drops only when it is explicitly non-active
    or explicitly superseded.

    Args:
        entry: A memory-entry dict.

    Returns:
        True if the entry should be retrievable, False otherwise.
    """
    return entry.get("status", "active") == "active" and not entry.get("superseded_by")


_STOPWORDS = frozenset(
    {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
        "was", "one", "our", "out", "day", "has", "had", "his", "how", "man",
        "new", "now", "old", "see", "two", "way", "who", "did", "its", "let",
        "put", "say", "she", "too", "use", "that", "this", "with", "from",
        "they", "them", "then", "than", "have", "will", "your", "what", "when",
        "were", "been", "into", "some", "more", "very", "just", "over", "only",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _significant_words(text: str) -> frozenset:
    """Lowercased alphanumeric tokens longer than 2 chars, minus stopwords."""
    return frozenset(
        w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS
    )


def _days_since(iso_str: str, now: datetime.datetime) -> float:
    """Whole-and-fractional days between ``now`` and an ISO-8601 timestamp.

    Returns 0.0 when the string is empty or unparseable, and clamps negative
    ages (timestamp in the future) to 0.0.
    """
    if not iso_str:
        return 0.0
    try:
        dt = datetime.datetime.fromisoformat(iso_str.rstrip("Z"))
    except (ValueError, AttributeError):
        return 0.0
    # Compare naive-to-naive: drop any tzinfo on both sides so a stored offset
    # cannot skew the age. Callers are expected to store UTC.
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    ref = now.replace(tzinfo=None) if now.tzinfo is not None else now
    return max(0.0, (ref - dt).total_seconds() / 86400.0)


def default_scorer(
    entry: Dict,
    context: Optional[Dict],
    *,
    now: Optional[datetime.datetime] = None,
) -> float:
    """Score one entry against retrieval context. Higher is more relevant.

    A simplified, self-contained re-expression of the production scorer. The
    score is a product of independent multipliers:

    * **base priority** — ``entry["base_priority"]`` (default 5).
    * **recency decay** — linear from the entry's ``created`` age against a
      90-day horizon, floored at 0.3. A same-day entry scores 1.0; a 90-day-old
      entry scores the 0.3 floor. ``now`` is injectable for deterministic tests.
    * **tag overlap** — fraction of the entry's tags that appear in
      ``context["tags"]``, applied as ``1 + overlap * 0.5``.
    * **project match** — x1.0 when the entry's project is None ("general",
      checked first so it stays neutral against any context), else x1.5 when
      projects match and x0.5 when they differ.
    * **message overlap** — fraction of the *message's* significant words that
      appear in the entry's content; when that fraction exceeds 0.15 it applies
      ``1 + overlap * 1.5``.

    Deliberately **omitted**: any access-count / retrieval-frequency term. In
    the production origin, retrieving an entry raised its access count, which
    raised its score, which got it retrieved again — a feedback loop that
    entrenched roughly 97% of the store as unreachable on neutral queries.
    Frequency couples ranking to persistence; it is left out on purpose.

    Args:
        entry: The memory-entry dict to score.
        context: Retrieval context; may be None. Recognized keys: ``project``
            (str | None), ``tags`` (Sequence[str]), ``message`` (str).
        now: Reference time for recency decay. Defaults to the current UTC time
            when None; pass a fixed value for reproducible results.

    Returns:
        A non-negative relevance score.
    """
    if context is None:
        context = {}
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    score = float(entry.get("base_priority", 5))

    # Recency: linear decay against a 90-day horizon, floored at 0.3.
    days = _days_since(entry.get("created") or "", now)
    recency = max(0.3, 1.0 - (days / 90.0) * 0.7)
    score *= recency

    # Tag overlap (topical).
    entry_tags = set(entry.get("tags") or [])
    context_tags = set(context.get("tags") or [])
    if entry_tags:
        overlap = len(entry_tags & context_tags) / len(entry_tags)
        score *= 1.0 + overlap * 0.5

    # Project match. A None entry-project ("general") is neutral regardless of
    # context — checked first so a general entry against a general context
    # scores x1.0, not the x1.5 same-project bonus.
    entry_project = entry.get("project")
    context_project = context.get("project")
    if entry_project is None:
        score *= 1.0
    elif entry_project == context_project:
        score *= 1.5
    else:
        score *= 0.5

    # Message-content word overlap.
    message = context.get("message") or ""
    if message:
        msg_words = _significant_words(message)
        ent_words = _significant_words(str(entry.get("content", "")))
        if msg_words and ent_words:
            overlap = len(ent_words & msg_words) / len(msg_words)
            if overlap > 0.15:
                score *= 1.0 + overlap * 1.5

    return score


# ── Retrieval ───────────────────────────────────────────────────────────────


def _entry_cost(entry: Dict, token_counter: Callable[[str], int]) -> int:
    """Token cost of an entry.

    Uses ``entry["token_count"]`` when it is present and a positive int;
    otherwise computes it from the content string. Never 0 for a missing count
    — a 0 default would silently exempt an entry from every budget (the
    production origin's bug).
    """
    tc = entry.get("token_count")
    if isinstance(tc, int) and not isinstance(tc, bool) and tc > 0:
        return tc
    return token_counter(str(entry.get("content", "")))


def retrieve(
    entries: Sequence[Dict],
    pool: Pool,
    context: Optional[Dict] = None,
    *,
    scorer: Callable[..., float] = default_scorer,
    token_counter: Callable[[str], int] = default_token_counter,
    is_active: Callable[[Dict], bool] = default_is_active,
    now: Optional[datetime.datetime] = None,
) -> List[Dict]:
    """Retrieve the entries a single pool admits, in render order.

    The algorithm, in order:

    1. **Filter** to active entries (``is_active``), then to the pool's types
       (``pool.types is None`` accepts every type).
    2. **Split** into always-scope entries (``scope == "always"``) and the
       rest (contextual).
    3. **Load every always-scope entry first**, bypassing *both*
       ``max_tokens`` and ``max_entries``. This is the invariant the library
       exists for: a behavioral correction can never be crowded out. (The
       production origin let always entries consume ``max_entries`` slots, so a
       burst of them could starve contextual retrieval — reported upstream as a
       bug; here they are exempt from both limits.)
    4. **Score** the contextual entries and sort by a fully deterministic key:
       ``(-score, created-or-"", id-or-"", original-index)``. Input order never
       changes the result except through that final ``original-index`` tiebreak,
       which breaks all remaining ties — so shuffling equal-key entries cannot
       change the output.
    5. **Greedily fill** the budget. An entry's cost is ``entry["token_count"]``
       when present and a positive int, else ``token_counter(content)`` — never
       0 for a missing count. An entry fits iff
       ``used + cost <= pool.max_tokens`` **and** the contextual selection still
       has room under ``max_entries``. On token-overflow the entry is skipped
       and scanning continues, because a smaller lower-scored entry may still
       fit. The entry-count cap applies to contextual entries only.

    The returned list holds the **same dict objects** as the input (no copies).
    Order is always-scope entries (in input order) followed by contextual
    entries (in sorted order). ``retrieve`` never mutates any entry.

    Args:
        entries: The candidate memory entries.
        pool: The reserved budget to fill.
        context: Retrieval context passed through to ``scorer``; may be None.
        scorer: Scoring callable ``(entry, context, *, now) -> float``.
        token_counter: Cost callable ``(text) -> int`` for entries lacking a
            positive ``token_count``.
        is_active: Predicate ``(entry) -> bool`` selecting live entries.
        now: Reference time forwarded to ``scorer`` for deterministic recency.

    Returns:
        The admitted entries in render order.
    """
    # 1. Filter.
    candidates = [e for e in entries if is_active(e)]
    if pool.types is not None:
        allowed = set(pool.types)
        candidates = [e for e in candidates if e.get("type", "note") in allowed]

    # 2. Split.
    always = [e for e in candidates if e.get("scope") == "always"]
    contextual = [e for e in candidates if e.get("scope") != "always"]

    # 3. Always-scope entries load unconditionally, in input order.
    results: List[Dict] = list(always)

    # 4. Score and deterministically sort the contextual entries.
    keyed = []
    for index, entry in enumerate(contextual):
        score = scorer(entry, context, now=now)
        keyed.append(
            (-score, entry.get("created") or "", entry.get("id") or "", index, entry)
        )
    keyed.sort(key=lambda k: k[:4])

    # 5. Greedy fill against the pool's token and entry budgets.
    used = 0
    selected = 0
    for _neg_score, _created, _id, _index, entry in keyed:
        if selected >= pool.max_entries:
            break
        cost = _entry_cost(entry, token_counter)
        if used + cost > pool.max_tokens:
            continue  # skip; a smaller later entry may still fit
        results.append(entry)
        used += cost
        selected += 1

    return results


def retrieve_pools(
    entries: Sequence[Dict],
    pools: Sequence[Pool],
    context: Optional[Dict] = None,
    *,
    scorer: Callable[..., float] = default_scorer,
    token_counter: Callable[[str], int] = default_token_counter,
    is_active: Callable[[Dict], bool] = default_is_active,
    now: Optional[datetime.datetime] = None,
) -> Dict[str, List[Dict]]:
    """Fill several pools from one entry set, without double-counting.

    Pools are processed in the given order. An entry selected by an earlier
    pool is excluded from every later pool, so each entry lands in at most one
    pool — the first (by pool order) that admits it. Dedupe is by object
    identity (``id()`` of the dict), so entries that lack an ``"id"`` key still
    dedupe correctly.

    Args:
        entries: The candidate memory entries.
        pools: The pools to fill, in priority order. Pool names must be unique.
        context: Retrieval context; may be None.
        scorer: See :func:`retrieve`.
        token_counter: See :func:`retrieve`.
        is_active: See :func:`retrieve`.
        now: See :func:`retrieve`.

    Returns:
        A dict mapping each ``pool.name`` to its admitted entries.

    Raises:
        ValueError: if two pools share a name.
    """
    seen_names = set()
    for pool in pools:
        if pool.name in seen_names:
            raise ValueError("duplicate pool name: %r" % (pool.name,))
        seen_names.add(pool.name)

    result: Dict[str, List[Dict]] = {}
    claimed = set()  # id() of every dict already placed in an earlier pool
    for pool in pools:
        available = [e for e in entries if id(e) not in claimed]
        selected = retrieve(
            available,
            pool,
            context,
            scorer=scorer,
            token_counter=token_counter,
            is_active=is_active,
            now=now,
        )
        result[pool.name] = selected
        for e in selected:
            claimed.add(id(e))
    return result
