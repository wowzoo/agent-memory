"""Neither tier may disappear from a response.

`_reserve_tier_slots` is a safety net: on the current corpus relevance ranking already
surfaces both tiers, so the paraphrase fixture scores identically with it on and off.
That is the desired outcome but it also means the fixture cannot prove the net works.
These tests exercise it directly, including the case it exists for -- one tier
outnumbering the other badly enough to sweep the whole window.
"""

from agent_memory.store import _reserve_tier_slots


def make(ranked_tiers):
    """(ranked list, rows) from a tier sequence, scores descending."""
    ranked = [(f"m{i}", 1.0 - i * 0.01) for i in range(len(ranked_tiers))]
    rows = {f"m{i}": {"tier": t} for i, t in enumerate(ranked_tiers)}
    return ranked, rows


def tiers(window, rows):
    return [rows[mid]["tier"] for mid, _ in window]


def test_cold_gets_slots_when_hot_sweeps_the_window():
    ranked, rows = make(["hot"] * 8 + ["cold"] * 4)
    window = _reserve_tier_slots(ranked, rows, limit=5, reserve=2)[:5]
    assert tiers(window, rows).count("cold") == 2
    assert tiers(window, rows).count("hot") == 3
    # Scores stay monotonically ordered within the window.
    scores = [s for _, s in window]
    assert scores == sorted(scores, reverse=True)


def test_hot_gets_slots_when_cold_sweeps_the_window():
    ranked, rows = make(["cold"] * 8 + ["hot"] * 3)
    window = _reserve_tier_slots(ranked, rows, limit=5, reserve=2)[:5]
    assert tiers(window, rows).count("hot") == 2


def test_untouched_when_both_tiers_already_present():
    ranked, rows = make(["hot", "cold", "hot", "cold", "hot", "cold"])
    assert _reserve_tier_slots(ranked, rows, limit=5, reserve=2)[:5] == ranked[:5]


def test_does_not_invent_rows_for_an_absent_tier():
    """A store with no hot files must still return a full window of cold results."""
    ranked, rows = make(["cold"] * 6)
    window = _reserve_tier_slots(ranked, rows, limit=5, reserve=2)[:5]
    assert tiers(window, rows) == ["cold"] * 5


def test_reserve_zero_is_pure_relevance():
    ranked, rows = make(["hot"] * 8 + ["cold"] * 4)
    assert _reserve_tier_slots(ranked, rows, limit=5, reserve=0) == ranked
