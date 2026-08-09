"""Korean-tuned sentence embeddings over onnxruntime.

Deliberately not fastembed: the model that actually works for this store
(dragonkue/multilingual-e5-small-ko) is not in fastembed's registry, and running
the ONNX graph directly removes both fastembed and torch from the dependency set.
"""

import threading
from typing import Any

import numpy as np

from . import config

_lock = threading.Lock()
_session: Any = None
_tokenizer: Any = None
_chunk_tokenizer: Any = None
_output_name: Any = None
_input_names: Any = None


def _load():
    """Return (tokenizer, session, output_name, input_names), loading once."""
    global _session, _tokenizer, _chunk_tokenizer, _output_name, _input_names
    if _session is not None:
        return _tokenizer, _session, _output_name, _input_names
    with _lock:
        if _session is not None:
            return _tokenizer, _session, _output_name, _input_names
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_file = config.MODEL_DIR / "model.onnx"
        tokenizer_file = config.MODEL_DIR / "tokenizer.json"
        if not model_file.exists() or not tokenizer_file.exists():
            raise FileNotFoundError(
                f"embedding model not found in {config.MODEL_DIR}. "
                f"Run scripts/export_model.py to create it, or set "
                f"AGENT_MEMORY_MODEL_DIR to an existing export."
            )

        # Two instances on purpose. Truncation and padding are properties of the
        # Tokenizer object, not per-call arguments, so a single shared instance
        # would silently truncate the chunker's own input to MAX_TOKENS *before*
        # it could window it -- measured at 42.9% of stored text reaching the
        # index, with a 4,954-token memory yielding 2 chunks instead of 14.
        tok = Tokenizer.from_file(str(tokenizer_file))
        tok.enable_truncation(max_length=config.MAX_TOKENS)
        tok.enable_padding()
        # Clear it explicitly rather than trusting the file: `save_pretrained`
        # bakes transformers' own truncation/padding defaults into tokenizer.json,
        # so a freshly exported tokenizer reintroduces exactly that bug (measured
        # on this store: 599 chunks collapse to 337). An export without those keys
        # is the lucky case, not the contract.
        _chunk_tokenizer = Tokenizer.from_file(str(tokenizer_file))
        _chunk_tokenizer.no_truncation()
        _chunk_tokenizer.no_padding()
        sess = ort.InferenceSession(
            str(model_file), providers=["CPUExecutionProvider"]
        )
        # Prefer the export's pooled output; fall back to mean-pooling the token
        # states, which is all a plain `optimum` export emits.
        names = {o.name for o in sess.get_outputs()}
        output = next(
            (
                n
                for n in ("sentence_embedding", "token_embeddings", "last_hidden_state")
                if n in names
            ),
            None,
        )
        if output is None:
            raise RuntimeError(f"unsupported ONNX outputs: {sorted(names)}")

        # Whether the graph wants `token_type_ids` depends on the optimum version
        # that produced it, not on the model: the checked-in export omits it, a
        # fresh one from optimum 2.x makes it a required input. Read it off the
        # session rather than assuming either shape.
        inputs = {i.name for i in sess.get_inputs()}

        # `_session` last: the fast path above gates on it without the lock, so a
        # reader must not be able to see a set `_session` alongside an unset
        # `_output_name` or `_input_names`. Live from mcp 2.x on, where tool
        # handlers run on worker threads rather than serialized on the event loop.
        _tokenizer, _output_name, _input_names, _session = tok, output, inputs, sess
    return _tokenizer, _session, _output_name, _input_names


def _encode(texts: list[str], prefix: str, batch_size: int = 16) -> np.ndarray:
    tokenizer, session, output, input_names = _load()
    out = []
    for start in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[start : start + batch_size]]
        encoded = tokenizer.encode_batch(batch)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        result = session.run([output], feed)[0]
        if result.ndim == 3:  # token states -> masked mean pool
            m = mask[:, :, None].astype(np.float32)
            result = (result * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        out.append(result.astype(np.float32))
    vectors = np.concatenate(out)
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)


def encode_passages(texts: list[str]) -> np.ndarray:
    return _encode(texts, config.PASSAGE_PREFIX)


def encode_query(text: str) -> np.ndarray:
    return _encode([text], config.QUERY_PREFIX)[0]


def chunk(text: str) -> list[str]:
    """Split into overlapping windows sized in *tokens*, not words.

    Korean runs ~1.9 characters per token under this XLM-R tokenizer, so a
    word-count budget mis-sizes chunks by a wide and text-dependent margin. The
    400/80 split here is the configuration the model comparison was measured
    under (438 chunks over 95 memories, 4.6 per memory).
    """
    _load()
    tokenizer = _chunk_tokenizer  # never truncating; see _load()
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if len(ids) <= config.CHUNK_TOKENS:
        return [text]
    step = config.CHUNK_TOKENS - config.CHUNK_OVERLAP
    chunks = []
    for start in range(0, len(ids), step):
        window = ids[start : start + config.CHUNK_TOKENS]
        if not window:
            break
        chunks.append(tokenizer.decode(window))
        if start + config.CHUNK_TOKENS >= len(ids):
            break
    return chunks


def passages(topic: str, content: str) -> list[str]:
    """The texts to embed for one memory: the topic on its own, then body chunks.

    Embedding the topic separately gives the memory's subject a vector that is not
    diluted by the 400 tokens of narrative sharing its chunk. Paraphrase queries
    ask about subjects, and measured on 20 such queries this lifted fused R@1 from
    20% to 25% on its own, and to 45% combined with topic-weighted BM25.
    """
    body = chunk(content)
    topic = topic.strip()
    if not topic or (len(body) == 1 and body[0].strip() == topic):
        return body
    return [topic, *body]


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
