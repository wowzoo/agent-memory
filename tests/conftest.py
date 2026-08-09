"""Shared test setup.

The store fixtures point `AGENT_MEMORY_HOME` at a tmp dir to isolate the database,
which also moves `MODEL_DIR` -- and a tmp dir holds no ONNX export. Running
test_graph.py or test_hot_reindex.py on their own therefore died on the missing
model; the full suite only passed because test_chunking.py loads the encoder first
and `embedding._load()` short-circuits on a warm session, so the tests that needed
a model never looked for one.

`config` is imported here, before any fixture reloads it, so `MODEL_DIR` still
holds the real export path. Isolate storage, share the encoder.
"""

import pytest

from agent_memory import config

REAL_MODEL_DIR = config.MODEL_DIR


@pytest.fixture
def real_model_dir(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_MODEL_DIR", str(REAL_MODEL_DIR))
    return REAL_MODEL_DIR
