#!/usr/bin/env python3
"""Encode the OR01-RET-ABS 0.2.0 noun hypernym interior.

This producer does not rank neighbors and does not revise the 0.1.0 vectors.
It uses the same EmbeddingGemma 2 text tower, prompt, surface, mean pool, and
L2 normalization as encode_saved_vectors.py, in batches small enough for the
local GPU. Query leaves, when a file is passed, are written beside the corpus
and are not rows of the interior file.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

SURFACE = "{member}: {definition}"
PROMPT_NAME = "SentenceSimilarity"
EXPECTED_PROMPT = "task: sentence similarity | query: "
REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
SNAPSHOT_ID = "or01-ret-abs-0.2.0-local-text"
QUERY_SNAPSHOT_ID = "or01-ret-abs-0.2.0-leaf-queries"
SPACE_ID = "local-text-768-cosine-l2-sentence-similarity"
DIMENSIONS = 768
PRIOR_ABSTRACT = "0096a2d4338d03ba94973d2679ad0604a0003958985208171acc13f2da535a81"
PRIOR_CONCRETE = "6c8e7bccf3b2e16cdb01abe1a80489d266b167954dfe035271ee30e8fe90e885"


def surface_of(members, definition):
    member = members[0] if members else ""
    gloss = definition[0] if definition else ""
    text = SURFACE.format(member=member, definition=gloss)
    return " ".join(text.split())


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def check_prior(path, expected):
    if not path.is_file():
        print(f"prior absent {path}", flush=True)
        return
    found = digest(path)
    if found != expected:
        raise SystemExit(f"0.1.0 hash changed for {path}: {found}")
    print(f"prior unchanged {path.name} {found}", flush=True)


def write_header(path, snapshot_id, count):
    path.write_text(
        "\n".join(
            [
                "mityane-saved-vectors-v1",
                f"space_id {SPACE_ID}",
                f"snapshot_id {snapshot_id}",
                "metric cosine",
                "normalization l2",
                f"dimensions {DIMENSIONS}",
                f"count {count}",
                "vectors",
                "",
            ]
        ),
        encoding="utf8",
    )


def append_hex_rows(vec_path, source_ids, matrix):
    with vec_path.open("a", encoding="utf8") as handle:
        for index, source_id in enumerate(source_ids):
            tokens = " ".join(float(value).hex() for value in matrix[index])
            handle.write(f"{source_id} {tokens}\n")
            if (index + 1) % 1000 == 0:
                print(f"wrote {index + 1} vector rows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prior-dir", default="")
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch size must be at least 1")
    if args.revision != REVISION:
        raise SystemExit("this encode is pinned to the 0.1.0 revision")

    snapshot_path = Path(args.snapshot)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf8"))
    if snapshot.get("binding_id") != "OR01-RET-ABS" or snapshot.get("revision") != "0.2.0":
        raise SystemExit("snapshot is not OR01-RET-ABS 0.2.0")
    rows = snapshot["abstract_corpus"]
    if len(rows) != 17191:
        raise SystemExit(f"expected 17,191 interior rows, found {len(rows)}")
    query_file = json.loads(Path(args.queries).read_text(encoding="utf8"))
    queries = query_file["queries"] if isinstance(query_file, dict) else query_file
    if not queries:
        raise SystemExit("query file is empty")
    interior_ids = {row["source_id"] for row in rows}
    for query in queries:
        if query["source_id"] in interior_ids:
            raise SystemExit(f"query {query['source_id']} is inside the interior")

    if args.prior_dir:
        prior = Path(args.prior_dir)
        check_prior(prior / "abstract.vec", PRIOR_ABSTRACT)
        check_prior(prior / "concrete.vec", PRIOR_CONCRETE)

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
    context_limit = int(getattr(config.text_config, "max_position_embeddings", 8192))

    def apply_prompt(text):
        if "{query}" in prompt:
            return prompt.replace("{query}", text)
        if "{}" in prompt:
            return prompt.replace("{}", text)
        return prompt + text

    def encode(texts, batch_size):
        encoded = []
        cursor = 0
        current = batch_size
        while cursor < len(texts):
            width = min(current, len(texts) - cursor)
            chunk = texts[cursor : cursor + width]
            prompted = [apply_prompt(text) for text in chunk]
            try:
                batch = tokenizer(prompted, padding=True, truncation=False, return_tensors="pt")
                if batch["input_ids"].shape[1] > context_limit:
                    raise SystemExit(f"a text exceeds the context limit {context_limit}")
                batch = {key: value.to("cuda") for key, value in batch.items() if key in ("input_ids", "attention_mask")}
                with torch.no_grad():
                    hidden = model(**batch).last_hidden_state
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1, dtype=torch.float32) / mask.sum(dim=1, dtype=torch.float32).clamp(min=1e-9)
                    pooled = torch.nn.functional.normalize(pooled.to(hidden.dtype), p=2, dim=1)
                if pooled.shape != (len(chunk), DIMENSIONS):
                    raise SystemExit(f"unexpected shape {tuple(pooled.shape)}")
                if not torch.isfinite(pooled).all():
                    raise SystemExit("non-finite embedding")
                encoded.append(pooled.float().cpu().numpy())
                cursor += width
            except RuntimeError as error:
                if "out of memory" not in str(error).lower() or current == 1:
                    raise
                torch.cuda.empty_cache()
                current = max(1, current // 2)
                print(f"batch reduced to {current}", flush=True)
        return np.concatenate(encoded, axis=0), current

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    identities = [row["source_id"] for row in rows]
    texts = [surface_of(row["members"], row["definition"]) for row in rows]
    floats_path = out / "abstract.f32"
    row_bytes = DIMENSIONS * 4
    done = 0
    if floats_path.is_file():
        size = floats_path.stat().st_size
        if size % row_bytes != 0:
            raise SystemExit(f"{floats_path} is not a whole number of rows")
        done = size // row_bytes
        if done > len(texts):
            raise SystemExit(f"{floats_path} has more rows than the snapshot")
    print(f"resume {done} of {len(texts)}", flush=True)
    started = time.time()
    batch_size = args.batch_size
    with floats_path.open("ab") as handle:
        while done < len(texts):
            width = min(batch_size, len(texts) - done)
            matrix, batch_size = encode(texts[done : done + width], batch_size)
            handle.write(matrix.astype(np.float32, copy=False).tobytes())
            handle.flush()
            done += width
            elapsed = time.time() - started
            print(f"encoded {done} {elapsed:.1f}s batch {batch_size}", flush=True)

    matrix = np.memmap(floats_path, dtype=np.float32, mode="r", shape=(len(texts), DIMENSIONS))
    vec_path = out / "abstract.vec"
    write_header(vec_path, SNAPSHOT_ID, len(texts))
    append_hex_rows(vec_path, identities, matrix)
    surfaces_path = out / "surfaces.tsv"
    surfaces_path.write_text(
        "".join(f"{source_id}\t{text}\n" for source_id, text in zip(identities, texts)),
        encoding="utf8",
    )

    query_ids = [query["source_id"] for query in queries]
    query_texts = [surface_of(query["members"], query["definition"]) for query in queries]
    query_matrix, batch_size = encode(query_texts, batch_size)
    query_floats = out / "queries.f32"
    query_floats.write_bytes(query_matrix.astype(np.float32, copy=False).tobytes())
    query_vec = out / "queries.vec"
    write_header(query_vec, QUERY_SNAPSHOT_ID, len(query_ids))
    append_hex_rows(query_vec, query_ids, query_matrix)
    query_surfaces = out / "query-surfaces.tsv"
    query_surfaces.write_text(
        "".join(f"{source_id}\t{text}\n" for source_id, text in zip(query_ids, query_texts)),
        encoding="utf8",
    )

    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    ref_file = cache / "hub" / "models--google--embeddinggemma-2" / "refs" / args.revision
    commit = ref_file.read_text(encoding="utf8").strip() if ref_file.is_file() else args.revision
    manifest = {
        "profile": "local-saved-vectors",
        "encoder_id": args.model,
        "encoder_revision": commit,
        "modalities": ["text"],
        "dtype": str(dtype).replace("torch.", ""),
        "device": "cuda",
        "loader": "transformers-text-tower",
        "prompt_name": PROMPT_NAME,
        "prompt": prompt,
        "pooling": "mean",
        "include_prompt": True,
        "surface": SURFACE,
        "dimensions": DIMENSIONS,
        "normalization": "l2",
        "metric": "cosine",
        "snapshot_id": SNAPSHOT_ID,
        "query_snapshot_id": QUERY_SNAPSHOT_ID,
        "interior_rows": len(texts),
        "query_rows": len(query_ids),
        "query_source_ids": query_ids,
        "snapshot_sha256": digest(snapshot_path),
        "files": {
            "abstract_f32": {"path": str(floats_path), "sha256": digest(floats_path)},
            "abstract_vec": {"path": str(vec_path), "sha256": digest(vec_path)},
            "surfaces": {"path": str(surfaces_path), "sha256": digest(surfaces_path)},
            "queries_f32": {"path": str(query_floats), "sha256": digest(query_floats)},
            "queries_vec": {"path": str(query_vec), "sha256": digest(query_vec)},
        },
        "admits_relevance_result": False,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(f"revision {commit}", flush=True)
    print(f"dtype {dtype}", flush=True)
    print(f"abstract {manifest['files']['abstract_vec']['sha256']}", flush=True)
    print(f"queries {manifest['files']['queries_vec']['sha256']}", flush=True)
    print("DONE encode", flush=True)


if __name__ == "__main__":
    main()
