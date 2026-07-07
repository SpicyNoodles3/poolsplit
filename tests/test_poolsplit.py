"""Deterministic tests for poolsplit. No wall-clock dependence: ``now`` injected."""

import datetime
import random

import pytest

from poolsplit import (
    Pool,
    default_is_active,
    default_scorer,
    default_token_counter,
    retrieve,
    retrieve_pools,
)

NOW = datetime.datetime(2026, 7, 7, tzinfo=datetime.timezone.utc)
FRESH = "2026-07-07T00:00:00Z"  # same day as NOW -> recency 1.0


def entry(**kw):
    """Build a memory entry, defaulting content so cost is well-defined."""
    kw.setdefault("content", "x")
    return kw


def ids(selected):
    return [e["id"] for e in selected]


# ── Budget / boundary ────────────────────────────────────────────────────────


def test_exact_fit_included():
    # cost == remaining budget -> included.
    e = entry(id="a", token_count=100)
    pool = Pool("p", max_tokens=100, max_entries=10)
    assert ids(retrieve([e], pool, now=NOW)) == ["a"]


def test_one_over_excluded():
    e = entry(id="a", token_count=101)
    pool = Pool("p", max_tokens=100, max_entries=10)
    assert ids(retrieve([e], pool, now=NOW)) == []


def test_skip_then_fit():
    # Big high-scored entry overflows; smaller lower-scored entry still fills.
    big = entry(id="big", token_count=90, base_priority=10, created=FRESH)
    small = entry(id="small", token_count=10, base_priority=1, created=FRESH)
    pool = Pool("p", max_tokens=50, max_entries=10)
    # big (score higher) is scanned first, overflows 50, is skipped; small fits.
    assert ids(retrieve([big, small], pool, now=NOW)) == ["small"]


def test_missing_token_count_is_computed_not_zero():
    # Regression pin: 400 chars of content must cost ~100 tokens, not 0.
    e = entry(id="a", content="c" * 400)  # 400 // 4 == 100
    fits = Pool("p", max_tokens=100, max_entries=10)
    too_small = Pool("q", max_tokens=99, max_entries=10)
    assert ids(retrieve([e], fits, now=NOW)) == ["a"]
    assert ids(retrieve([e], too_small, now=NOW)) == []  # would be included if cost were 0


def test_token_count_present_wins_over_counter():
    # Long content but a small declared token_count -> the declared count is used.
    e = entry(id="a", content="c" * 4000, token_count=5)
    pool = Pool("p", max_tokens=10, max_entries=10)
    assert ids(retrieve([e], pool, now=NOW)) == ["a"]


def test_zero_token_count_falls_back_to_counter():
    # token_count present but not a positive int -> counter is used.
    e = entry(id="a", content="c" * 400, token_count=0)
    too_small = Pool("q", max_tokens=99, max_entries=10)
    assert ids(retrieve([e], too_small, now=NOW)) == []  # 100-token cost, not 0


def test_max_entries_cap():
    es = [entry(id=str(i), token_count=1, base_priority=i, created=FRESH) for i in range(5)]
    pool = Pool("p", max_tokens=1000, max_entries=3)
    assert len(retrieve(es, pool, now=NOW)) == 3


def test_max_entries_counts_contextual_only():
    always = [entry(id="A", scope="always", token_count=1),
              entry(id="B", scope="always", token_count=1)]
    ctx = entry(id="c", token_count=1)
    pool = Pool("p", max_tokens=1000, max_entries=1)
    got = retrieve(always + [ctx], pool, now=NOW)
    assert set(ids(got)) == {"A", "B", "c"}  # 2 always + 1 contextual == 3


# ── Always-scope invariants ──────────────────────────────────────────────────


def test_always_exceeds_budget_still_returned():
    e = entry(id="a", scope="always", token_count=10_000)
    pool = Pool("p", max_tokens=1, max_entries=1)
    assert ids(retrieve([e], pool, now=NOW)) == ["a"]


def test_always_dont_consume_entry_slots():
    always = [entry(id="A", scope="always"), entry(id="B", scope="always")]
    ctx = entry(id="c", token_count=1)
    pool = Pool("p", max_tokens=1000, max_entries=1)
    got = retrieve(always + [ctx], pool, now=NOW)
    assert len(got) == 3
    assert ids(got)[:2] == ["A", "B"]  # always first, input order


def test_always_wrong_type_not_returned_by_typed_pool():
    e = entry(id="a", scope="always", type="note")
    pool = Pool("p", max_tokens=1000, max_entries=10, types=("correction",))
    assert ids(retrieve([e], pool, now=NOW)) == []


def test_max_tokens_zero_only_always():
    always = entry(id="A", scope="always")
    ctx = entry(id="c", token_count=1)
    pool = Pool("p", max_tokens=0, max_entries=10)
    assert ids(retrieve([always, ctx], pool, now=NOW)) == ["A"]


def test_max_entries_zero_only_always():
    always = entry(id="A", scope="always")
    ctx = entry(id="c", token_count=1)
    pool = Pool("p", max_tokens=1000, max_entries=0)
    assert ids(retrieve([always, ctx], pool, now=NOW)) == ["A"]


# ── Determinism ──────────────────────────────────────────────────────────────


def test_equal_score_sorts_created_then_id_then_index():
    # Identical scores (same base_priority, no created -> recency floor for all,
    # but give them distinct created to test the created key first).
    a = entry(id="z", base_priority=5, created="2026-07-01T00:00:00Z")
    b = entry(id="a", base_priority=5, created="2026-07-01T00:00:00Z")
    # Same created and same score -> id breaks the tie: "a" before "z".
    pool = Pool("p", max_tokens=1000, max_entries=10)
    got = retrieve([a, b], pool, now=NOW)
    assert ids(got) == ["a", "z"]


def test_equal_everything_uses_original_index():
    # No id, no created, identical score -> original index is the final tiebreak.
    a = entry(base_priority=5)
    b = entry(base_priority=5)
    a["content"], b["content"] = "first", "second"
    pool = Pool("p", max_tokens=1000, max_entries=10)
    got = retrieve([a, b], pool, now=NOW)
    assert [e["content"] for e in got] == ["first", "second"]


def test_shuffle_distinct_scores_identical_output():
    es = [entry(id=str(i), base_priority=i, created=FRESH) for i in range(20)]
    pool = Pool("p", max_tokens=10_000, max_entries=100)
    baseline = ids(retrieve(list(es), pool, now=NOW))
    for seed in range(5):
        shuffled = list(es)
        random.Random(seed).shuffle(shuffled)
        assert ids(retrieve(shuffled, pool, now=NOW)) == baseline


# ── Scorer ───────────────────────────────────────────────────────────────────


def test_recency_decay_floor():
    old = entry(id="old", created="2000-01-01T00:00:00Z")
    # 90-day horizon long exceeded -> floor 0.3 * base_priority 5 == 1.5.
    assert default_scorer(old, {}, now=NOW) == pytest.approx(5 * 0.3)


def test_fresh_scores_full_recency():
    fresh = entry(id="new", created=FRESH)
    assert default_scorer(fresh, {}, now=NOW) == pytest.approx(5.0)


def test_project_mismatch_penalty():
    same = entry(project="alpha", created=FRESH)
    diff = entry(project="beta", created=FRESH)
    none = entry(project=None, created=FRESH)
    ctx = {"project": "alpha"}
    assert default_scorer(same, ctx, now=NOW) == pytest.approx(5 * 1.5)
    assert default_scorer(diff, ctx, now=NOW) == pytest.approx(5 * 0.5)
    assert default_scorer(none, ctx, now=NOW) == pytest.approx(5 * 1.0)


def test_tag_overlap_boost():
    e = entry(tags=["retrieval", "memory"], created=FRESH)
    no_ctx = default_scorer(e, {"tags": []}, now=NOW)
    with_ctx = default_scorer(e, {"tags": ["retrieval"]}, now=NOW)
    # overlap 1/2 -> x(1 + 0.5*0.5) == x1.25.
    assert with_ctx == pytest.approx(no_ctx * 1.25)


def test_message_overlap_boost():
    e = entry(content="rotate the exposed api key today", created=FRESH)
    base = default_scorer(e, {"message": ""}, now=NOW)
    boosted = default_scorer(e, {"message": "rotate exposed api key"}, now=NOW)
    assert boosted > base


def test_message_overlap_below_threshold_no_boost():
    e = entry(content="rotate key", created=FRESH)
    base = default_scorer(e, {"message": ""}, now=NOW)
    # overlap = |ent & msg| / |msg significant words|. One of the entry's two
    # words appears among nine message words -> 1/9 ~= 0.11 < 0.15, no boost.
    msg = "rotate alpha bravo charlie delta echo foxtrot golf hotel"
    same = default_scorer(e, {"message": msg}, now=NOW)
    assert same == pytest.approx(base)


def test_now_injection_changes_score():
    e = entry(created="2026-01-01T00:00:00Z")
    early = default_scorer(e, {}, now=datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc))
    later = default_scorer(e, {}, now=datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc))
    assert early > later  # older relative to a later "now" -> lower score


# ── retrieve_pools ───────────────────────────────────────────────────────────


def test_dedupe_first_pool_wins():
    e = entry(id="shared", type="correction", token_count=1, created=FRESH)
    pools = [
        Pool("first", max_tokens=100, max_entries=10, types=("correction",)),
        Pool("second", max_tokens=100, max_entries=10, types=("correction",)),
    ]
    got = retrieve_pools([e], pools, now=NOW)
    assert ids(got["first"]) == ["shared"]
    assert ids(got["second"]) == []


def test_disjoint_pools_dont_interact():
    corr = entry(id="c", type="correction", token_count=1, created=FRESH)
    task = entry(id="t", type="task", token_count=1, created=FRESH)
    pools = [
        Pool("corr", max_tokens=100, max_entries=10, types=("correction",)),
        Pool("task", max_tokens=100, max_entries=10, types=("task",)),
    ]
    got = retrieve_pools([corr, task], pools, now=NOW)
    assert ids(got["corr"]) == ["c"]
    assert ids(got["task"]) == ["t"]


def test_dedupe_without_id_key():
    # Two type-overlapping pools, entries with no "id" -> id()-based dedupe.
    e = entry(type="correction", token_count=1, created=FRESH)
    e["content"] = "only one"
    pools = [
        Pool("first", max_tokens=100, max_entries=10, types=("correction",)),
        Pool("second", max_tokens=100, max_entries=10, types=("correction",)),
    ]
    got = retrieve_pools([e], pools, now=NOW)
    assert len(got["first"]) == 1
    assert len(got["second"]) == 0


def test_duplicate_pool_names_raise():
    pools = [Pool("dup", max_tokens=10, max_entries=1), Pool("dup", max_tokens=10, max_entries=1)]
    with pytest.raises(ValueError):
        retrieve_pools([], pools, now=NOW)


def test_empty_pools_list():
    assert retrieve_pools([entry(id="a")], [], now=NOW) == {}


def test_empty_entries_all_pools_empty():
    pools = [Pool("a", max_tokens=10, max_entries=1), Pool("b", max_tokens=10, max_entries=1)]
    got = retrieve_pools([], pools, now=NOW)
    assert got == {"a": [], "b": []}


# ── Filtering ────────────────────────────────────────────────────────────────


def test_superseded_excluded():
    e = entry(id="a", superseded_by="b")
    pool = Pool("p", max_tokens=100, max_entries=10)
    assert ids(retrieve([e], pool, now=NOW)) == []


def test_archived_status_excluded():
    e = entry(id="a", status="archived")
    pool = Pool("p", max_tokens=100, max_entries=10)
    assert ids(retrieve([e], pool, now=NOW)) == []


def test_missing_status_included():
    e = entry(id="a")  # no status key
    pool = Pool("p", max_tokens=100, max_entries=10)
    assert ids(retrieve([e], pool, now=NOW)) == ["a"]


def test_default_is_active_semantics():
    assert default_is_active({}) is True
    assert default_is_active({"status": "active"}) is True
    assert default_is_active({"status": "archived"}) is False
    assert default_is_active({"superseded_by": "x"}) is False
    assert default_is_active({"status": "active", "superseded_by": "x"}) is False


# ── Validation ───────────────────────────────────────────────────────────────


def test_negative_max_tokens_raises():
    with pytest.raises(ValueError):
        Pool("p", max_tokens=-1, max_entries=1)


def test_negative_max_entries_raises():
    with pytest.raises(ValueError):
        Pool("p", max_tokens=1, max_entries=-1)


# ── No mutation / identity ───────────────────────────────────────────────────


def test_retrieve_does_not_mutate_entries():
    e = entry(id="a", token_count=1, created=FRESH)
    before = dict(e)
    pool = Pool("p", max_tokens=100, max_entries=10)
    retrieve([e], pool, now=NOW)
    assert e == before  # no access-count bump, no last_accessed


def test_returned_entries_are_same_objects():
    e = entry(id="a", token_count=1, created=FRESH)
    pool = Pool("p", max_tokens=100, max_entries=10)
    got = retrieve([e], pool, now=NOW)
    assert got[0] is e


# ── default_token_counter ────────────────────────────────────────────────────


def test_token_counter_never_zero():
    assert default_token_counter("") == 1
    assert default_token_counter("abc") == 1  # 3 // 4 == 0 -> floored to 1
    assert default_token_counter("c" * 400) == 100
