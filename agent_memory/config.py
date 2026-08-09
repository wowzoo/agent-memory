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
# So: rank everything on relevance alone, then guarantee each tier a couple of slots.
# Relevance decides the order; this only decides that neither tier disappears.
MIN_TIER_SLOTS = _int("AGENT_MEMORY_MIN_TIER_SLOTS", 2)

# Set to "0"/"false" to stop mirroring Claude Code memory files entirely.
HOT_ENABLED = os.environ.get("AGENT_MEMORY_HOT", "1").lower() not in ("0", "false", "no")
# Half-life in days for recency weighting; 0 disables it.
HALF_LIFE_DAYS = _int("AGENT_MEMORY_HALF_LIFE_DAYS", 0)
