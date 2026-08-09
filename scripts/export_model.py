"""Export the Korean embedding model to ONNX.

HuggingFace ships dragonkue/multilingual-e5-small-ko as safetensors only, so the
runtime artifact has to be produced locally. This keeps that step reproducible
rather than depending on a pre-built file of unknown provenance.

    pip install -e '.[export]'
    python scripts/export_model.py

Writes model.onnx + tokenizer.json into $AGENT_MEMORY_MODEL_DIR and verifies the
export against the PyTorch reference before declaring success.

The export emits `last_hidden_state`; agent_memory.embedding mean-pools it with
the attention mask, matching this model's sentence-transformers pooling config.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_memory import config  # noqa: E402

REPO = "dragonkue/multilingual-e5-small-ko"
PROBES = [
    "query: 한국어 검색이 왜 안 됐지",
    "passage: 임베딩 모델이 영어 전용이어서 한국어 유사도가 뭉개졌다.",
    "passage: AI-DLC v2 2.5.59 세대 4트리 재적용을 완료했다.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--out", type=Path, default=config.MODEL_DIR)
    args = parser.parse_args()

    import numpy as np
    import torch
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoModel, AutoTokenizer

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"exporting {args.repo} -> {args.out}")

    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    tokenizer.save_pretrained(args.out)
    ORTModelForFeatureExtraction.from_pretrained(args.repo, export=True).save_pretrained(args.out)

    # optimum writes model.onnx; keep the flat layout agent_memory.embedding expects
    produced = next(args.out.rglob("model.onnx"))
    target = args.out / "model.onnx"
    if produced != target:
        produced.replace(target)

    def mean_pool(states, mask):
        m = mask.unsqueeze(-1).float()
        return (states * m).sum(1) / m.sum(1).clamp(min=1e-9)

    reference_model = AutoModel.from_pretrained(args.repo).eval()
    batch = tokenizer(PROBES, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        reference = mean_pool(reference_model(**batch).last_hidden_state, batch["attention_mask"])
    reference = torch.nn.functional.normalize(reference, dim=1).numpy()

    # Re-import through the runtime path so the check covers our own loader too.
    from agent_memory import embedding

    embedding._session = None  # force a reload from the freshly written files
    actual = embedding._encode(PROBES, prefix="")

    sims = (reference * actual).sum(axis=1)
    print("cosine vs PyTorch reference:", np.round(sims, 5))
    if sims.min() < 0.999:
        raise SystemExit(f"export mismatch: min cosine {sims.min():.5f} < 0.999")
    print(f"ok — model.onnx and tokenizer.json written to {args.out}")


if __name__ == "__main__":
    main()
