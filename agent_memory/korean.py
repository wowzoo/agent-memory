"""Korean lexical analysis for the keyword retrieval lane.

Why this module exists at all: SQLite FTS5 has no tokenizer that handles Korean.
Measured on 95 real Korean memories, against LIKE-substring ground truth:

    query   truth   'porter ascii' (MARM)   trigram   Kiwi morphemes
    패치      33          21 (64%)           0 (0%)      33 (100%)
    검증      75          56 (75%)           0 (0%)      75 (100%)
    커밋      80          77 (96%)           0 (0%)      79 (99%)
    메모리    20          15 (75%)          20 (100%)    19 (95%)

`porter ascii` and `unicode61` split on whitespace, so a query term never matches
the same word carrying a different particle ("패치" vs "패치를"). `trigram` fixes
substrings but cannot match anything shorter than 3 characters -- and two-syllable
words are the common case in Korean, so it scores a flat 0% on 패치/검증/커밋.

So we tokenize with Kiwi ourselves and feed FTS5 a normalized token stream.
"""

import os
import re
import threading

# Content-bearing parts of speech. Particles (J*), verb endings (E*), affixes (X*)
# and punctuation (S[FEPO]) are dropped -- stripping those is the whole point.
_KEEP_TAGS = frozenset(
    {
        "NNG",  # common noun
        "NNP",  # proper noun
        "NNB",  # bound noun
        "NR",  # numeral
        "SL",  # foreign/Latin word
        "SN",  # number
        "SH",  # Chinese character
        "VV",  # verb stem (lemmatized: 고쳤다 -> 고치)
        "VA",  # adjective stem
        "MAG",  # general adverb
        "W_SERIAL",  # version strings: "2.5.59"
        "W_URL",
        "W_EMAIL",
    }
)

# Korean function-like nouns and adverbs that survive the POS filter but carry no
# retrievable sense. Kept deliberately short: the POS filter already removes the
# bulk of the noise, and BM25 discounts frequent terms by inverse document
# frequency, so an over-long list costs recall for no precision gain.
_BASE_STOPWORDS = frozenset(
    """
    것 수 등 및 때 점 경우 통해 대해 위해 관련 이후 이전 자체 정도 부분 상태
    사용 진행 확인 필요 결과 내용 방식 경우 이번 해당 다시 아직 매우 아주
    """.split()
)

_STOPWORDS = _BASE_STOPWORDS | {
    w for w in os.environ.get("AGENT_MEMORY_EXTRA_STOPWORDS", "").split() if w
}

# Latin/number runs are indexed verbatim alongside morphemes so identifiers keep
# working ("aidlc", "v2", "2.5.59").
_WORD_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]*|[가-힣]+")

_kiwi = None
_kiwi_lock = threading.Lock()


def _analyzer():
    """Lazy singleton. Kiwi loads a ~30MB model, so never at import time."""
    global _kiwi
    if _kiwi is None:
        with _kiwi_lock:
            if _kiwi is None:
                from kiwipiepy import Kiwi

                _kiwi = Kiwi()
    return _kiwi


def _keep(token: str) -> bool:
    return len(token) > 1 and token not in _STOPWORDS


def tokens(text: str) -> list[str]:
    """Morphemes plus surface forms, deduplicated, order preserved.

    Both are indexed because Kiwi strips derivational prefixes: "재적용" analyses
    to 재/XPN + 적용/NNG, so a morpheme-only index turns a search for "재적용"
    into a search for "적용" -- measured as 44 hits where only 28 documents
    contain the compound. Keeping the surface form gives the exact compound its
    own term, so BM25 ranks a true "재적용" match above a bare "적용" one while
    the morpheme still provides the loose match.
    """
    out: dict[str, None] = {}
    for tok in _analyzer().tokenize(text):
        if tok.tag in _KEEP_TAGS:
            form = tok.form.lower()
            if _keep(form):
                out[form] = None
    for word in _WORD_RE.findall(text):
        word = word.lower()
        if _keep(word):
            out[word] = None
    return list(out)


def index_text(text: str) -> str:
    """The string stored in the FTS5 column."""
    return " ".join(tokens(text))


def match_query(text: str) -> str | None:
    """Build an FTS5 MATCH expression from user input.

    OR rather than AND: a natural-language Korean question rarely shares every
    term with the memory that answers it, and this lane only has to produce a
    candidate pool that the semantic lane then reranks. Each term is quoted so
    user input cannot be parsed as FTS5 syntax (NEAR, *, ^, column filters).
    """
    terms = tokens(text)
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)
