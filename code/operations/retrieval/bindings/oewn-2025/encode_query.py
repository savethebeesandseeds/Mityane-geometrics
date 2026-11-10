#!/usr/bin/env python3
"""Encode one or more query surfaces with the 0.2.0 text tower.

Each line is encoded as-is under the sentence-similarity prompt. The corpus
file is not rewritten. Output is a float32 matrix and a surfaces tsv beside it.
"""

import argparse
import json
import os
from pathlib import Path

EXPECTED_PROMPT = "task: sentence similarity | query: "
PROMPT_NAME = "SentenceSimilarity"
REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
DIMENSIONS = 768


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default="examples")
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--file", default="")
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    texts = list(args.text)
    if args.file:
        texts.extend(
            line.strip()
            for line in Path(args.file).read_text(encoding="utf8").splitlines()
            if line.strip() and not line.startswith("#")
        )
    if not texts:
        raise SystemExit("pass --text or --file")
    if args.revision != REVISION:
        raise SystemExit("this encode is pinned to the 0.1.0 revision")

    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this local encode")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    config.vision_config = None
    config.audio_config = None
    model = AutoModel.from_pretrained(args.model, revision=args.revision, config=config, dtype=dtype)
    model.to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prompt_path = hf_hub_download(args.model, "config_sentence_transformers.json", revision=args.revision)
    prompt = json.loads(Path(prompt_path).read_text(encoding="utf8"))["prompts"][PROMPT_NAME]
    if prompt != EXPECTED_PROMPT:
        raise SystemExit(f"prompt is {prompt!r}")

    prompted = [prompt.replace("{query}", text) if "{query}" in prompt else prompt + text for text in texts]
    batch = tokenizer(prompted, padding=True, truncation=False, return_tensors="pt")
    batch = {key: value.to("cuda") for key, value in batch.items() if key in ("input_ids", "attention_mask")}
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1, dtype=torch.float32) / mask.sum(dim=1, dtype=torch.float32).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled.to(hidden.dtype), p=2, dim=1)
    if pooled.shape != (len(texts), DIMENSIONS) or not torch.isfinite(pooled).all():
        raise SystemExit("bad embedding")
    matrix = pooled.float().cpu().numpy().astype(np.float32, copy=False)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.f32").write_bytes(matrix.tobytes())
    (out / f"{args.name}.tsv").write_text(
        "".join(f"q{index}\t{text}\n" for index, text in enumerate(texts)),
        encoding="utf8",
    )
    print(f"encoded {len(texts)} {out / (args.name + '.f32')}", flush=True)


if __name__ == "__main__":
    main()
