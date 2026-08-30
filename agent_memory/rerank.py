"""Cross-encoder reranking of the fused candidates. On behind the gate.

Measured on 72 paraphrase queries (cold, one store state, top 10 rescored):
R@1 72.2% -> 91.7%, R@5 94.4% -> 98.6%, MRR 0.819 -> 0.943. That R@5 is the
depth-10 ceiling -- of the answers the base lanes place in the top 10, this puts all
but one in the top 5. The Korean finetune is what makes it work: the multilingual
`bge-reranker-base` had to be blended into the fusion to win at all (+2 cases), while
`Dongjin-kr/ko-reranker` wins by *replacing* the fused order outright, which is also
why there is no weight to overfit here.

It costs about 300 ms per candidate on CPU -- 3 s per recall at depth 10 against roughly
10 ms for the lanes alone -- so it was opt-in until 2026-08-30. What changed is
`store._corroborated`: reranking every recall measured worse than reranking none, and
reranking only the recalls whose top hit one lane never retrieved beats both. That is
about one in six, so the mean bill is ~0.5 s. `rerank=True` forces it past the gate;
`AGENT_MEMORY_RERANK=0` turns it off entirely.

It also has a shape of question it loses on. Each candidate is judged in one pass over
`topic + body` truncated to RERANK_MAX_TOKENS, so evidence past that window is invisible
to it: on `eval_recall.py`, whose queries are sentences lifted verbatim from the middle
of a memory, R@5 falls from 99.2% to 79.5% -- and 72 of those 132 query sentences sit
outside the window. Paraphrase queries ask what a memory is *about*, which its opening
answers, which is why the same model gains 19 points there. Scoring every chunk and
keeping the maximum would close the gap at four to five inferences per candidate.
"""

import threading
from typing import Any

import numpy as np

from . import config

_lock = threading.Lock()
_session: Any = None
_tokenizer: Any = None
_input_names: Any = None


def available() -> bool:
    """Whether an export is present, without loading half a gigabyte to find out."""
    return (config.RERANK_MODEL_DIR / "model.onnx").exists() and (
        config.RERANK_MODEL_DIR / "tokenizer.json"
    ).exists()


def _load():
    global _session, _tokenizer, _input_names
    if _session is not None:
        return _tokenizer, _session, _input_names
    with _lock:
        if _session is not None:
            return _tokenizer, _session, _input_names
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_file = config.RERANK_MODEL_DIR / "model.onnx"
        tokenizer_file = config.RERANK_MODEL_DIR / "tokenizer.json"
        if not model_file.exists() or not tokenizer_file.exists():
            raise FileNotFoundError(
                f"reranker not found in {config.RERANK_MODEL_DIR}. Run "
                f"scripts/export_reranker.py, or set AGENT_MEMORY_RERANK_MODEL_DIR."
            )

        tok = Tokenizer.from_file(str(tokenizer_file))
        # State both explicitly rather than trusting the file: `save_pretrained`
        # bakes transformers' defaults into tokenizer.json, and inheriting them is
        # how a re-export once silently truncated indexing input.
        tok.no_truncation()
        tok.no_padding()
        tok.enable_truncation(max_length=config.RERANK_MAX_TOKENS)
        tok.enable_padding()
        sess = ort.InferenceSession(
            str(model_file), providers=["CPUExecutionProvider"]
        )
        # `_session` last: the fast path above gates on it without the lock.
        _tokenizer, _input_names, _session = tok, {i.name for i in sess.get_inputs()}, sess
    return _tokenizer, _session, _input_names


def scores(query: str, passages: list[str], batch_size: int = 16) -> list[float]:
    """Relevance logit per passage. Unbounded and comparable only within one call."""
    tokenizer, session, input_names = _load()
    out: list[float] = []
    for start in range(0, len(passages), batch_size):
        encoded = tokenizer.encode_batch(
            [(query, p) for p in passages[start : start + batch_size]]
        )
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        feed = {"input_ids": ids}
        if "attention_mask" in input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        # Pair encoding fills type_ids for models that want them; XLM-R does not.
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        out.extend(session.run(None, feed)[0].reshape(-1).tolist())
    return out
