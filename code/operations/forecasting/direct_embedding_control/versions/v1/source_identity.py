#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../../../LICENSE.
"""Pin the exact control implementation closure before compiling inside Linux."""
import argparse
import hashlib
import pathlib
import struct

FILES = sorted((
    "code/operations/forecasting/direct_embedding_control/versions/v1/include/cuwacunu/networks/direct_embedding_control/v1/control.h",
    "code/operations/forecasting/direct_embedding_control/versions/v1/include/cuwacunu/networks/direct_embedding_control/v1/artifact.h",
    "code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/sidecar.h",
    "code/shared/artifacts/versions/v1/include/cuwacunu/artifacts/v1/tensor_state.h",
    "code/operations/forecasting/evaluation/versions/v1/include/cuwacunu/evaluation/latent_dynamics/v1/baselines.h",
    "code/operations/forecasting/evaluation/versions/v1/src/baselines.cpp",
))


def identity(root):
    fields = []
    for name in FILES:
        actual = (root / name).resolve(strict=True)
        actual.relative_to(root)
        if not actual.is_file():
            raise ValueError(f"Not an implementation source: {name}")
        fields.extend((name, hashlib.sha256(actual.read_bytes()).hexdigest()))
    digest = hashlib.sha256()
    for value in ("cuwacunu_artifact_record_v1", "source_closure_v1", str(len(fields)), *fields):
        raw = value.encode("utf-8")
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(identity(pathlib.Path(args.root).resolve(strict=True)))
