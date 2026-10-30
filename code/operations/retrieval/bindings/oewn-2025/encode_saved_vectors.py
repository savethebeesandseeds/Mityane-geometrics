#!/usr/bin/env python3
"""Write saved vectors for one local text encoder.

The retrieval ranker never imports this file. Any other encoder can write the
same mityane-saved-vectors-v1 text. This producer is the EmbeddingGemma 2
text-only instance used for the OR01-RET-ABS check.

The checkpoint's sentence-transformers entry loads a video processor, and that
processor imports torchvision. This file leaves vision_config and audio_config
unset, applies the published SentenceSimilarity prompt, mean-pools every token
including that prompt, and L2-normalizes. That is the text path in the
checkpoint's sentence-transformers modules.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


SURFACE = "{member}: {definition}"
PROMPT_NAME = "SentenceSimilarity"


def surface(members, definition):
    member = members[0] if members else ""
    gloss = definition[0] if definition else ""
    return SURFACE.format(member=member, definition=gloss)


def write_vectors(path, binding, rows):
    lines = [
        "mityane-saved-vectors-v1",
        f"space_id {binding['space_id']}",
        f"snapshot_id {binding['snapshot_id']}",
        f"metric {binding['metric']}",
        f"normalization {binding['normalization']}",
        f"dimensions {binding['dimensions']}",
        f"count {len(rows)}",
        "vectors",
    ]
    for source_id, values in rows:
        encoded = " ".join(float(value).hex() for value in values)
        lines.append(f"{source_id} {encoded}")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf8"))
    if snapshot.get("binding_id") != "OR01-RET-ABS":
        raise SystemExit("snapshot is not OR01-RET-ABS")
    texts = []
    identities = []
    for row in snapshot["concrete_catalog"]:
        identities.append(("concrete", row["source_id"]))
        texts.append(surface(row["members"], row["definition"]))
    for row in snapshot["abstract_corpus"]:
        identities.append(("abstract", row["source_id"]))
        texts.append(surface(row["members"], row["definition"]))

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
    prompt_path = hf_hub_download(
        args.model, "config_sentence_transformers.json", revision=args.revision
    )
    prompt = json.loads(Path(prompt_path).read_text(encoding="utf8"))["prompts"][PROMPT_NAME]
    context_limit = int(getattr(config.text_config, "max_position_embeddings", 8192))

    def apply_prompt(text):
        if "{query}" in prompt:
            return prompt.replace("{query}", text)
        if "{}" in prompt:
            return prompt.replace("{}", text)
        return prompt + text

    prompted = [apply_prompt(text) for text in texts]
    batch = tokenizer(prompted, padding=True, truncation=False, return_tensors="pt")
    if batch["input_ids"].shape[1] > context_limit:
        raise SystemExit(f"a text exceeds the context limit {context_limit}")
    batch = {key: value.to("cuda") for key, value in batch.items() if key in ("input_ids", "attention_mask")}
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1, dtype=torch.float32) / mask.sum(dim=1, dtype=torch.float32).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled.to(hidden.dtype), p=2, dim=1)
    if pooled.shape != (len(texts), 768):
        raise SystemExit(f"unexpected shape {tuple(pooled.shape)}")
    if not torch.isfinite(pooled).all():
        raise SystemExit("non-finite embedding")
    values = pooled.float().cpu().numpy()

    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    model_cache = cache / "hub" / "models--google--embeddinggemma-2"
    ref_file = model_cache / "refs" / args.revision
    snapshot_dir = model_cache / "snapshots"
    if ref_file.is_file():
        commit = ref_file.read_text(encoding="utf8").strip()
    else:
        commits = sorted(path.name for path in snapshot_dir.iterdir() if path.is_dir()) if snapshot_dir.is_dir() else []
        commit = commits[-1] if commits else args.revision
    name_or_path = getattr(model, "name_or_path", None) or args.model

    binding = {
        "space_id": "local-text-768-cosine-l2-sentence-similarity",
        "snapshot_id": "or01-ret-abs-0.1.0-local-text",
        "metric": "cosine",
        "normalization": "l2",
        "dimensions": 768,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    concrete = [(source_id, values[index]) for index, (kind, source_id) in enumerate(identities) if kind == "concrete"]
    abstract = [(source_id, values[index]) for index, (kind, source_id) in enumerate(identities) if kind == "abstract"]
    concrete_path = out / "concrete.vec"
    abstract_path = out / "abstract.vec"
    write_vectors(concrete_path, binding, concrete)
    write_vectors(abstract_path, binding, abstract)

    queries = []
    for query in snapshot["queries"]:
        if not query["eligible_for_task_association"]:
            continue
        ids = query["concrete_source_ids"]
        labels = " ".join(query["relevant_source_ids"])
        queries.append(f"point {ids[0]} 29 label {labels}")
        if len(ids) == 2:
            queries.append(f"difference {ids[0]} {ids[1]} 29 label {labels}")
        queries.append("shared " + " ".join(ids) + f" 29 label {labels}")
    query_path = out / "queries.txt"
    query_path.write_text("\n".join(queries) + "\n", encoding="utf8")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "profile": "local-saved-vectors",
        "encoder_id": args.model,
        "encoder_revision": commit,
        "encoder_path": name_or_path,
        "modalities": ["text"],
        "dtype": str(dtype).replace("torch.", ""),
        "device": "cuda",
        "loader": "transformers-text-tower",
        "prompt_name": PROMPT_NAME,
        "prompt": prompt,
        "pooling": "mean",
        "include_prompt": True,
        "surface": SURFACE,
        "dimensions": 768,
        "normalization": "l2",
        "metric": "cosine",
        "task_role": "sentence-similarity",
        "texts": [
            {"population": kind, "source_id": source_id, "surface": text, "prompted": prompted[index]}
            for index, ((kind, source_id), text) in enumerate(zip(identities, texts))
        ],
        "files": {
            "concrete": {"path": str(concrete_path), "sha256": digest(concrete_path)},
            "abstract": {"path": str(abstract_path), "sha256": digest(abstract_path)},
            "queries": {"path": str(query_path), "sha256": digest(query_path)},
        },
        "snapshot_sha256": hashlib.sha256(Path(args.snapshot).read_bytes()).hexdigest(),
        "admits_relevance_result": False,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(f"revision {commit}")
    print(f"dtype {dtype}")
    print(f"concrete {digest(concrete_path)}")
    print(f"abstract {digest(abstract_path)}")
    print(f"manifest {manifest_path}")


if __name__ == "__main__":
    main()
