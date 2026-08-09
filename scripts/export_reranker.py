"""Export a cross-encoder reranker to ONNX, the same way export_model.py does.

Same reason as the embedding export: the runtime carries onnxruntime and tokenizers,
not torch, so the artifact has to be produced locally. Exporting every candidate the
same way also keeps a model comparison from measuring export differences instead of
models -- BAAI ships a prebuilt ONNX and the Korean finetune does not, and taking one
of each would confound exactly the question being asked.

    pip install -e '.[export]'
    python scripts/export_reranker.py --repo Dongjin-kr/ko-reranker --out ~/.agent-memory/models/ko-reranker

Writes model.onnx + tokenizer.json and verifies the logits against the PyTorch
reference before declaring success.
"""

import argparse
from pathlib import Path

# A relevant and an irrelevant passage for the same question: the export is only
# useful if it keeps them apart, so the check reads the gap, not just the values.
PROBES = [
    ("한국어 검색이 왜 안 됐지", "임베딩 모델이 영어 전용이어서 한국어 유사도가 뭉개졌다."),
    ("한국어 검색이 왜 안 됐지", "슬라이드 애니메이션 spid 와 도형 생성 순서를 맞췄다."),
    ("평가셋을 몇 건으로 늘렸나", "패러프레이즈 질의를 20건에서 72건으로 확장했다."),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.expanduser()

    import numpy as np
    import torch
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    print(f"exporting {args.repo} -> {out}")

    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    tokenizer.save_pretrained(out)
    ORTModelForSequenceClassification.from_pretrained(
        args.repo, export=True
    ).save_pretrained(out)

    produced = next(out.rglob("model.onnx"))
    target = out / "model.onnx"
    if produced != target:
        produced.replace(target)

    # tokenizer.json arrives with transformers' defaults baked in, including
    # truncation.max_length. The embedding loader had to override that after a
    # re-export silently cut inputs; the reranker loader sets its own limit per call
    # for the same reason, so leave the file alone and note it here.
    queries = [q for q, _ in PROBES]
    passages = [p for _, p in PROBES]
    batch = tokenizer(queries, passages, padding=True, truncation=True,
                      return_tensors="pt")
    reference_model = AutoModelForSequenceClassification.from_pretrained(args.repo).eval()
    with torch.no_grad():
        reference = reference_model(**batch).logits.squeeze(-1).numpy()

    import onnxruntime as ort
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    names = {i.name for i in session.get_inputs()}
    feed = {k: v.numpy() for k, v in batch.items() if k in names}
    actual = session.run(None, feed)[0].squeeze(-1)

    print("reference logits:", np.round(reference, 4))
    print("onnx logits     :", np.round(actual, 4))
    drift = np.abs(reference - actual).max()
    print(f"max abs drift   : {drift:.6f}")
    if drift > 1e-3:
        raise SystemExit(f"export mismatch: drift {drift:.6f} > 1e-3")
    if not (actual[0] > actual[1]):
        raise SystemExit("export is not discriminative: relevant scored below irrelevant")
    print(f"ok — model.onnx and tokenizer.json written to {out}")


if __name__ == "__main__":
    main()
