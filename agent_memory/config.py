"""Environment-driven settings.

Every retrieval-shaping value is overridable. MARM hardcoded its embedding model
(`config/settings.py:198`), which is why its store could not be moved off an
English-only encoder without editing source -- the single defect that motivated
this project.
"""

import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


HOME = Path(os.environ.get("AGENT_MEMORY_HOME", Path.home() / ".agent-memory"))
DB_PATH = Path(os.environ.get("AGENT_MEMORY_DB", HOME / "memory.db"))

# --- Embedding model -------------------------------------------------------
# Default is dragonkue/multilingual-e5-small-ko, exported to ONNX.
# Measured on 95 real Korean memories (known-item retrieval, identical chunks):
#   multilingual-e5-small-ko   R@1 71.6%  R@5 93.7%  MRR 0.820
#   jina-v2-small-en (MARM)    R@1 31.6%  R@5 65.3%  MRR 0.462
#   paraphrase-ml-MiniLM       R@1 29.5%  R@5 53.7%  MRR 0.415
# Swap in a larger Korean retriever (e.g. nlpai-lab/KURE-v1, 1024-dim) by pointing
# MODEL_DIR at its export and setting EMBED_DIM to match.
MODEL_DIR = Path(
    os.environ.get(
        "AGENT_MEMORY_MODEL_DIR", HOME / "models" / "multilingual-e5-small-ko"
    )
)
MODEL_NAME = os.environ.get("AGENT_MEMORY_MODEL_NAME", "multilingual-e5-small-ko")
EMBED_DIM = _int("AGENT_MEMORY_EMBED_DIM", 384)

# e5-family models are trained with asymmetric prefixes and lose accuracy without
# them. Set both to "" for a model that does not use them.
QUERY_PREFIX = os.environ.get("AGENT_MEMORY_QUERY_PREFIX", "query: ")
PASSAGE_PREFIX = os.environ.get("AGENT_MEMORY_PASSAGE_PREFIX", "passage: ")

# --- Chunking --------------------------------------------------------------
# Token counts, not word counts: Korean runs ~1.9 chars/token under the XLM-R
# tokenizer, so a word-based budget (MARM's approach) mis-sizes chunks badly.
# 400/80 is the configuration the model selection was measured under.
MAX_TOKENS = _int("AGENT_MEMORY_MAX_TOKENS", 512)
CHUNK_TOKENS = _int("AGENT_MEMORY_CHUNK_TOKENS", 400)
CHUNK_OVERLAP = _int("AGENT_MEMORY_CHUNK_OVERLAP", 80)

# --- Retrieval -------------------------------------------------------------
RECALL_LIMIT = _int("AGENT_MEMORY_RECALL_LIMIT", 5)
CANDIDATE_POOL = _int("AGENT_MEMORY_CANDIDATE_POOL", 50)
# RRF's k decides how much *corroboration* outweighs *conviction*. The usual 60 comes
# from TREC-scale runs over pools of thousands; against 106 memories it flattens the
# curve so far that two mediocre lane hits beat one excellent one -- a memory ranked
# 25th and 26th in the two lanes outscored one ranked 3rd in a single lane, which is
# how a semantic top-3 hit missing from the BM25 pool ended up 28th overall. Lowering
# k sharpens the head. Measured (72 paraphrase queries, cold-only): R@1 70.8% -> 73.6%,
# MRR 0.809 -> 0.823, R@5 unchanged; and on the verbatim eval_recall.py fixture
# R@5 99.2% -> 100%, MRR 0.918 -> 0.923. Not knife-edge: every k in 5..20 beats 60,
# and 10 is where R@5 stops improving before k=5 starts costing it a case.
RRF_K = _int("AGENT_MEMORY_RRF_K", 10)


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# bm25() column weights for (topic, body). Measured on 20 paraphrase queries:
# 3x/1x took the fused lane from R@1 20.0% / MRR 0.437 to 45.0% / 0.599 together
# with the topic vector below.
TOPIC_WEIGHT = _float("AGENT_MEMORY_TOPIC_WEIGHT", 3.0)
BODY_WEIGHT = _float("AGENT_MEMORY_BODY_WEIGHT", 1.0)

# Weight of the concept-graph co-citation lane in the fusion; 0 disables it, and it
# only runs in hybrid mode. Kept well under 1.0 because the lane is noisy on its own
# (R@1 26% unaided) -- it is worth having as a tie-breaker, not as a peer of BM25 and
# the embeddings. Measured (72 paraphrase queries, one store state, cold-only ->
# both tiers): R@5 94.4 -> 95.8% and 93.1 -> 95.8%, R@1 and MRR no worse in either.
# Raised 0.5 -> 0.75 on 2026-08-21 after the store roughly doubled (cold 124 -> 203,
# hot 693 -> 895 sections, relations 1372 -> 1833). The optimum moved with it: 0.75 now
# beats 0.5 on all six numbers with nothing traded away (cold 66.7/86.1/0.763 ->
# 68.1/87.5/0.773, both tiers 59.7/76.4/0.683 -> 61.1/76.4/0.695), and the old note
# that "above 0.5 R@1 erodes" no longer holds -- 0.65..1.0 now *gain* two R@5 cases.
# w=1.0 is the only variant lifting R@5 in both modes but it sells a cold R@1 case, so
# it was not taken: this lane is only ever adopted when the gain is free.
# The lane's contribution grew as the corpus got noisier, which is the whole reason to
# re-sweep on growth: both-tier R@1 over a graph-off baseline went +0 (adoption) ->
# +4.1pp (2026-08-12) -> +13.9pp here (47.2 -> 61.1). Re-sweep when the corpus jumps.
GRAPH_LANE_WEIGHT = _float("AGENT_MEMORY_GRAPH_LANE_WEIGHT", 0.75)
# Base-lane hits used as co-citation seeds. Feedback quality falls off fast: these are
# the memories assumed relevant before anything has confirmed they are.
GRAPH_LANE_SEEDS = _int("AGENT_MEMORY_GRAPH_LANE_SEEDS", 3)

# --- Reranking ---------------------------------------------------------------
# A cross-encoder rescoring of the fused top RERANK_DEPTH. Measured on 72 paraphrase
# queries (cold, one state): R@1 72.2% -> 91.7%, R@5 94.4% -> 98.6%, MRR 0.819 ->
# 0.943, where 98.6% is the depth-10 ceiling. It costs ~300 ms per candidate on CPU --
# 3 s per recall against ~10 ms for the lanes alone -- which is why it was opt-in until
# the gate existed. `rerank=True` still forces it for one question, gate and all.
# Depth 10 rather than 20: 20 raises the ceiling to 100% but doubles the bill for the
# one remaining case. See agent_memory/rerank.py and scripts/experiment_rerank.py.
# On since 2026-08-30, and only because RERANK_GATE makes it safe: ungated, reranking
# everything measured *worse* than reranking nothing (pooled R@1 55.9% vs 57.6%, R@5
# 75.3% vs 82.9%). Gated it runs for about one recall in six, so the mean cost is around
# 0.5 s rather than 3 s. Set to 0 for a latency-bound caller.
# ⚠️ Single-lane modes are never gated -- one lane cannot corroborate itself -- so
# `mode="lexical"` and `mode="semantic"` pay the full 3 s on every call with this on.
# They are diagnostic paths; pass `rerank=False` when sweeping them.
RERANK_ENABLED = os.environ.get("AGENT_MEMORY_RERANK", "1").lower() in ("1", "true", "yes")
RERANK_MODEL_DIR = Path(
    os.environ.get("AGENT_MEMORY_RERANK_MODEL_DIR", HOME / "models" / "ko-reranker-int8")
)
RERANK_DEPTH = _int("AGENT_MEMORY_RERANK_DEPTH", 10)
RERANK_MAX_TOKENS = _int("AGENT_MEMORY_RERANK_MAX_TOKENS", 512)

# Skip the reranker when every lane already retrieved the top hit. Only applies to
# reranking that comes from RERANK_ENABLED: a caller passing `rerank=True` asked for
# the cross-encoder and gets it.
#
# This is what makes RERANK_ENABLED=1 safe to set. Measured on 170 cases (cold 72 +
# hot 38 + 60 middle-sentence queries, 2026-08-29): reranking everything is *worse*
# than reranking nothing (pooled R@1 55.9% vs 57.6%, R@5 75.3% vs 82.9%), because it
# wins on the two authored fixtures and destroys verbatim mid-document lookups
# (R@1 81.7% -> 33.3%). Gated, it beats the un-reranked baseline on R@1 and R@5 on
# all three fixtures at 18% of the latency bill -- pooled R@1 57.6 -> 64.1%,
# R@5 82.9 -> 85.3%, mean latency about 0.5 s instead of 3 s.
# ⚠️ It deliberately leaves value on the table for hot-tier questions: 23.7 -> 36.8%
# R@1 where always-on reaches 71.1%. A caller asking about a memory file's contents
# should still pass `rerank=True` for that question.
#
# 🔴 Re-measured 2026-09-13 on a corpus twice that size, with the fixtures rebuilt and
# the conditions made identical (same window, `ORDER BY id` so the verbatim slice is
# fixed, gate path rather than off): **the 81.7 -> 33.3% collapse above does not
# reproduce.** Forcing the reranker now *raises* R@1 on all three fixtures -- cold
# 59.7 -> 83.3, hot 36.8 -> 57.9, verbatim 51.7 -> 66.7, pooled 51.8 -> 71.8% -- and
# pooled R@5 with it (85.9 -> 88.3%). So the accuracy case for this gate is gone.
# It stays on for the other half of the trade: measured the same day, the reranker costs
# 4.6-6.5 s, not the ~3 s above, and cold queries are skipped 100% of the time, so they
# answer in 0.06 s today. Ungating makes them 100x slower. ⇒ the open question is not
# on/off, it is the cost curve: RERANK_DEPTH below 10 (never measured -- only 10 vs 20),
# and relaxing this from three corroborating lanes to two. Until that is measured, three
# lanes is not known to be the right condition; it is only known to cost 20%p of R@1.
# See `docs/measurements.md` `### 정리 후 기준선`.
RERANK_GATE = os.environ.get("AGENT_MEMORY_RERANK_GATE", "1").lower() not in (
    "0",
    "false",
    "no",
)

# Give a base lane's own first choice a slot when no other lane retrieved it at all.
#
# Diagnosed 2026-08-30 (`scripts/diagnose_candidates.py`): of the cold answers the
# fusion puts outside the top 5, only 3 of 72 were missing from both lanes' pools --
# candidate generation is not the leak. The rest were retrieved, often by a lane that
# was certain: one hot query is the lexical lane's *first* hit and eighth after fusion.
# That is RRF working as designed, one lane at rank 2 scoring 1/(k+2) against two lanes
# at rank 8 scoring 2/(k+8), and lowering RRF_K to 10 did not reach these cases.
#
# Categorical on purpose, like the tier slots and the rerank gate: it promotes to the
# last slot of the caller's window and there is no weight to tune. A score multiplier
# would need one, and this repo has rejected two of those already.
#
# Measured at RECALL_LIMIT=5 -- the only window where it can act -- on a store copy with
# this session's own record sections removed, because those sections quote the fixture's
# cases and the lexical lane then retrieves the record instead of the answer:
#   cold 72   55.6 / 79.2 unchanged
#   hot 38    R@5 73.7% -> 76.3%, R@1 unchanged, nothing worse
# One case, and it is exactly the case the diagnosis named. Lowering RRF_K to 5 buys the
# same hot R@5 by rewarding conviction everywhere, and pays cold R@5 2.8 points for it;
# this buys it only where a lane stands alone.
# On since 2026-08-30: it cost nothing measurable and no case anywhere got worse.
RESCUE_LANE_LEADERS = os.environ.get(
    "AGENT_MEMORY_RESCUE_LANE_LEADERS", "1"
).lower() in ("1", "true", "yes")

# Slots reserved for a tier that relevance ranking would otherwise shut out. 0 = off.
#
# Both halves of this number were measured. Hot files outnumber the archive six to
# one here (582 sections vs 101 memories), so left alone the archive stops being
# reachable -- and the archive is precisely where the long-form detail deliberately
# moved out of the files lives. Two things were tried and rejected:
#
#   - a flat score multiplier for hot: dropped the fixture's cold targets to
#     hybrid R@1 0% / R@5 40% (from 55% / 90% cold-only), because it promoted files
#     even for questions the files do not answer.
#   - strict tier interleaving: recovered R@5 to 80% but pinned R@1 at 0% by always
#     spending slot 1 on a hot row, and cost MRR (0.394 vs 0.520 for one pool).
#
# So: rank everything on relevance alone, then guarantee each tier a slot.
# Relevance decides the order; this only decides that neither tier disappears.
#
# One slot rather than two since 2026-08-30, when the number itself was measured for the
# first time -- and in the window a caller actually gets, which is the only one where it
# acts at all. At `limit=5`, 0, 1 and 2 are identical on the cold fixture and 2 costs one
# hot_over_cold case, because the reranker (on by default since the same day) already
# recovers the two cold rows the second slot used to buy. With `AGENT_MEMORY_RERANK=0`
# those two come back, so 2 is right for a caller who turns reranking off. Above 2 the
# guarantees contradict each other in a five-row window and the hot pass, running second,
# wins: reserving 3 puts three cold answers out of the window.
MIN_TIER_SLOTS = _int("AGENT_MEMORY_MIN_TIER_SLOTS", 1)

# Set to "0"/"false" to stop mirroring Claude Code memory files entirely.
HOT_ENABLED = os.environ.get("AGENT_MEMORY_HOT", "1").lower() not in ("0", "false", "no")
# Half-life in days for recency weighting; 0 disables it.
HALF_LIFE_DAYS = _int("AGENT_MEMORY_HALF_LIFE_DAYS", 0)
