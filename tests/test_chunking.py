"""Chunking must not silently drop text.

The regression this guards: `enable_truncation(512)` on the tokenizer shared with
the chunker made chunk() window an already-truncated input, so a 4,954-token
memory produced 2 chunks instead of 14 and only 42.9% of the store's text ever
reached the semantic index. Nothing failed loudly -- recall just quietly got worse
on long memories.
"""

from tokenizers import Tokenizer

from agent_memory import config, embedding

KOREAN_PARAGRAPH = (
    "AI-DLC v2 2.5.59 세대 4트리 완결 및 상류 이슈 전수 정리 작업을 수행했다. "
    "사용자가 다른 모델로 패치한 트리를 검증하고 doctor 결과를 확인했다. "
)


def _raw_tokens(text: str) -> int:
    tok = Tokenizer.from_file(str(config.MODEL_DIR / "tokenizer.json"))
    # The measuring instrument must not truncate either: an export produced by
    # `save_pretrained` carries truncation in tokenizer.json, which would cap this
    # count at MAX_TOKENS and make the fixture look too short to test anything.
    tok.no_truncation()
    return len(tok.encode(text, add_special_tokens=False).ids)


def test_short_text_is_one_chunk():
    assert embedding.chunk("한국어 검색을 고쳤다.") == ["한국어 검색을 고쳤다."]


def test_long_text_is_fully_covered():
    text = KOREAN_PARAGRAPH * 60
    total = _raw_tokens(text)
    assert total > config.MAX_TOKENS * 4, "fixture must exceed the model window severalfold"

    chunks = embedding.chunk(text)

    # With overlap, the chunks re-cover some tokens, so the sum exceeds the
    # original count. The failure mode being guarded is the opposite: chunk text
    # summing to a fraction of the input because the tail was truncated away.
    covered = sum(_raw_tokens(c) for c in chunks)
    assert covered >= total, f"chunks cover only {covered}/{total} tokens"

    expected = 1 + -(-max(0, total - config.CHUNK_TOKENS) // (config.CHUNK_TOKENS - config.CHUNK_OVERLAP))
    assert len(chunks) == expected, f"{len(chunks)} chunks, expected {expected}"


def test_every_chunk_fits_the_model_window():
    chunks = embedding.chunk(KOREAN_PARAGRAPH * 60)
    for c in chunks:
        assert _raw_tokens(c) <= config.MAX_TOKENS
