#!/usr/bin/env python3
# Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
"""Create synthetic CACHE88 admission inputs; never build or execute a model.

Run only in the documented Linux container:
  python3 code/latent_market/versions/v1/tests/make_capture_fixture.py NEW_LEAF

NEW_LEAF must be beneath /opt/cuwacunu/build. The existing runtime-selected CPB
checkpoint is read and SHA256 checked; it is never copied, modified or trained.
The separate prepare CLI may subsequently use this card for a synthetic CUDA
correctness witness. This generator itself makes no Encoder calls.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys


ROOT = Path("/cuwacunu")
BUILD_ROOT = Path("/opt/cuwacunu/build")
CACHE88 = struct.Struct("<q5dq4d")
INTERVAL_MS = 60_000
ROWS = 600
LATENCY_MS = 5
# Declared historical provenance bound, supplied by the retained experiment
# record. This is not a new fit or a timestamp measured from checkpoint bytes.
ENCODER_FIT_CUTOFF_MS = 1_705_881_599_999
BASE_MS = int(dt.datetime(2027, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def new_file(path: Path, content: str | bytes) -> None:
    payload = content.encode("utf-8") if isinstance(content, str) else content
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def read_block(path: Path, block_name: str) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(?m)#.*$|//.*$", "", text)
    matches = list(re.finditer(r"\b" + re.escape(block_name) + r"\s*\{([^{}]*)\}", text, re.S))
    require(len(matches) == 1, f"Exactly one {block_name} block required: {path}")
    fields: dict[str, str] = {}
    for item in matches[0].group(1).split(";"):
        if not item.strip():
            continue
        match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*?)\s*", item, re.S)
        require(match is not None, f"Malformed {block_name} field in {path}")
        assert match is not None
        key, value = match.groups()
        require(key not in fields and value != "", f"Duplicate/empty {key} in {path}")
        fields[key] = value
    return fields


def layout_tests() -> int:
    """Bind the struct to the actual cache declaration, including trade F64."""
    header = ROOT / "code/data/include/cuwacunu/data/record_types.h"
    text = header.read_text(encoding="utf-8")
    match = re.search(r"struct kline_cache_t\s*\{(.*?)\n\};", text, re.S)
    require(match is not None, "Actual kline_cache_t declaration missing")
    assert match is not None
    values = match.group(1).split("/* Values */")[-1]
    declarations = re.findall(r"\b(ms_t|double|int32_t)\s+([A-Za-z_][A-Za-z_0-9]*)\s*;", values)
    expected = [
        ("ms_t", "open_time"), ("double", "open_price"), ("double", "high_price"),
        ("double", "low_price"), ("double", "close_price"), ("double", "volume"),
        ("ms_t", "close_time"), ("double", "quote_asset_volume"),
        ("double", "number_of_trades"), ("double", "taker_buy_base_volume"),
        ("double", "taker_buy_quote_volume"),
    ]
    require(declarations == expected, "CACHE88 fields/types changed; revise this version explicitly")
    require(CACHE88.size == 88, "CACHE88 must contain exactly 88 little-endian packed bytes")
    sample = (1000, 100.0, 102.0, 99.0, 101.0, 12.0, 60999, 1212.0, 7.0, 5.0, 505.0)
    packed = CACHE88.pack(*sample)
    require(len(packed) == 88 and CACHE88.unpack(packed) == sample, "CACHE88 pack/unpack roundtrip")
    require(struct.unpack_from("<q", packed, 48)[0] == 60999, "close-time key must be at byte offset 48")
    require(struct.unpack_from("<d", packed, 64)[0] == 7.0, "trade count must remain F64 in the cache")
    require(len(packed * ROWS) == ROWS * 88, "600-record exact byte length")
    return 6


def endpoint(tick: int) -> int:
    return BASE_MS + (tick + 1) * INTERVAL_MS - 1


def source_bytes(channel: int) -> bytes:
    records = bytearray()
    for tick in range(ROWS):
        phase = 0.13 * tick + 0.47 * channel
        level = 100.0 + 17.0 * channel + 0.003 * tick
        open_price = level + 0.10 * math.sin(phase - 0.04)
        close_price = level + 0.10 * math.sin(phase)
        high_price = max(open_price, close_price) + 0.03
        low_price = min(open_price, close_price) - 0.03
        volume = 10.0 + 2.0 * channel + 0.2 * (1.0 + math.cos(phase * 0.7))
        quote_volume = volume * close_price
        trade_count = float(20 + channel * 3 + tick % 7)
        taker_base = volume * (0.45 + 0.02 * math.sin(phase * 0.3))
        taker_quote = taker_base * close_price
        row = (
            BASE_MS + tick * INTERVAL_MS,
            open_price, high_price, low_price, close_price, volume,
            endpoint(tick), quote_volume, trade_count, taker_base, taker_quote,
        )
        require(all(math.isfinite(value) for value in row), "Synthetic fields must all be finite")
        require(low_price > 0 and low_price <= open_price <= high_price and
                low_price <= close_price <= high_price and volume > 0 and trade_count > 0,
                "Synthetic positive observed OHLC/activity contract")
        records.extend(CACHE88.pack(*row))
    require(len(records) == ROWS * CACHE88.size, "Raw cache byte length mismatch")
    # Validate every exact close key and validity sentinel after serialization.
    for tick in range(ROWS):
        row = CACHE88.unpack_from(records, tick * CACHE88.size)
        require(row[0] != -(1 << 63) and row[6] == endpoint(tick), "Raw observed key/validity roundtrip")
    return bytes(records)


def write_manifest(destination: Path, paths: list[Path], comment: str) -> None:
    unique = sorted({path.resolve(strict=True) for path in paths}, key=str)
    require(unique, "Manifest must not be empty")
    lines = [f"# {comment}\n"]
    for path in unique:
        require(path.is_file() and "\n" not in str(path), "Manifest requires regular single-line paths")
        lines.append(f"{sha256(path)}  {path}\n")
    new_file(destination, "".join(lines))


def make_fixture(leaf: Path) -> None:
    runtime_path = ROOT / "config/encoder_runtime.conf"
    settings_path = ROOT / "config/encoder.conf"
    runtime = read_block(runtime_path, "ENCODER_RUNTIME")
    settings = read_block(settings_path, "ENCODER")
    require(runtime["MODEL_TAG"] == settings["MODEL_TAG"] == "CPB-v1", "Runtime must select the actual CPB-v1")
    require((int(settings["CHANNEL_COUNT"]), int(settings["HISTORY_LENGTH"]), int(settings["INPUT_WIDTH"])) == (3, 128, 9),
            "Fixture requires the actual saved C3/H128/F9 CPB geometry")
    require(float(settings["SAMPLING_INTERVAL"]) == INTERVAL_MS and settings["DEVICE"].startswith("cuda"),
            "Fixture requires actual minute CUDA config")
    channel_ids = [int(item.strip()) for item in settings["CHANNEL_IDS"].split(",")]
    require(len(channel_ids) == 3 and len(set(channel_ids)) == 3, "Three distinct actual semantic channels required")
    checkpoint = (runtime_path.parent / runtime["CHECKPOINT"]).resolve(strict=True)
    checkpoint_hash = sha256(checkpoint)
    require(re.fullmatch(r"[0-9a-f]{64}", runtime["CHECKPOINT_SHA256"]) is not None and
            checkpoint_hash == runtime["CHECKPOINT_SHA256"], "Actual checkpoint bytes disagree with runtime SHA256")

    sources: list[Path] = []
    for channel in range(3):
        path = leaf / f"synthetic-channel-{channel}.cache.bin"
        new_file(path, source_bytes(channel))
        sources.append(path)
    require(len({sha256(path) for path in sources}) == 3, "Three distinct synthetic channel values required")

    origins = leaf / "origins.tsv"
    origin_rows = [
        ("synthetic-train", 150, "train", "synthetic-train-block"),
        ("synthetic-validation", 450, "validation", "synthetic-validation-block"),
        ("synthetic-missing-boundary", 599, "validation", "synthetic-invalid-block"),
    ]
    new_file(origins, "# pair_id endpoint_ms decision_ms role dependence_group\n" + "".join(
        f"{name} {endpoint(tick)} {endpoint(tick)+LATENCY_MS} {role} {group}\n"
        for name, tick, role, group in origin_rows))

    card = leaf / "synthetic-capture-card.txt"
    new_file(card, f"""SYNTHETIC CACHE88 -> EXISTING CPB-v1 CAPTURE CORRECTNESS CARD v1
This is a synthetic engineering witness, without market inputs or quality scores.
The generator creates data and pins files only; it never executes or trains a model.
The separate preparation CLI may make eval/NoGrad CUDA captures with zero updates,
zero scaler/normalizer fits, no old measurement replay, no publication or promotion.
Runtime selection: {runtime['MODEL_TAG']} / {runtime['INSTANCE']}.
Checkpoint: {checkpoint}
Checkpoint SHA256 independently verified against config/encoder_runtime.conf: {checkpoint_hash}
Encoder settings: {settings_path}, SHA256 {sha256(settings_path)}.
Historical Encoder fit cutoff declared as {ENCODER_FIT_CUTOFF_MS} (2024-01-21 23:59:59.999 UTC).
That bound is inherited provenance asserted for this card, not a new fit or inferred
from checkpoint bytes. Synthetic observations start in January 2027 after that bound.
Geometry C3/H128/F9 from actual config; semantic channel IDs ordered {channel_ids}.
Three distinct raw CACHE88 files, 600 complete one-minute candles each; no CSV or
normalized cache, all finite positive OHLC and activity, all rows observed.
Packed little endian <q5dq4d: 88 bytes/record, close_time offset48, trade-count F64 offset64.
Source ordering SOURCE_0/1/2 binds generated channels to semantic IDs in config order.
Close endpoint of tick j is UTC2027-01-01T00:00:00 + (j+1)*60000ms - 1ms.
Available/decision time endpoint+{LATENCY_MS}ms is an explicit simulation assumption,
not measured live receipt latency. No execution-price or portfolio assumption.
TRAIN raw ticks0..299; VALIDATION raw ticks300..599; no TEST or confirmation role.
K4 s1; latent horizons1,16,128; future-risk R128, complete-window eligibility.
Valid TRAIN origin tick150 has raw envelope ticks20..278 wholly in TRAIN.
Valid VALIDATION origin tick450 has raw envelope ticks320..578 wholly in VALIDATION.
Invalid VALIDATION origin tick599 retains missing future associations and boundary
exclusion (envelope469..727); no interpolation, row shifting or missing-target zero.
Required native endpoints21, available native windows18, capture each once.
Expected pair ledger3 origins x3 horizons:6 eligible,3 invalid. All source values/masks
remain origin-side or target-side according to the exact timestamp association.
Future-risk labels: adjacent close log returns j1..128, centered population deviation
in one-minute log-return units, and max loss relative to origin close; no annualization.
Acceptance is pipeline correctness, finite native arrays, unchanged Encoder/scaler,
explicit invalid retention and immutable roundtrip. No latent quality, state-learning,
investment, source-data selection or generalization claim follows from this fixture.
""")

    closure_sources: list[Path] = []
    for relative in (
        "code/data", "code/encoders/causal_patch_bottleneck_mae/versions/v1",
        "code/latent_market/versions/v1", "code/config/include", "code/integration/include", "config",
    ):
        directory = ROOT / relative
        require(directory.is_dir(), f"Required source closure directory absent: {directory}")
        closure_sources.extend(path for path in directory.rglob("*") if path.is_file() and not path.is_symlink())
    closure_sources.extend([ROOT / "Makefile", ROOT / "code/scripts/task.sh"])
    closure = leaf / "source-closure.sha256"
    write_manifest(closure, closure_sources, "Synthetic capture source dependency superset; retained local SDK is environmental")
    pins = leaf / "inputs.sha256"
    write_manifest(pins, sources + [origins, card, runtime_path, settings_path, checkpoint, closure],
                   "Independent byte pins for this synthetic capture correctness witness")

    fields = {
        "EXPERIMENT_ID": "SYNTHETIC-CPB-V1-CACHE88-CORRECTNESS",
        "CARD": str(card), "CARD_SHA256": sha256(card),
        "PINS": str(pins), "PINS_SHA256": sha256(pins),
        "SOURCE_CLOSURE": str(closure), "SOURCE_CLOSURE_SHA256": sha256(closure),
        "ORIGINS": str(origins), "ORIGINS_SHA256": sha256(origins),
        "ENCODER_SETTINGS": str(settings_path), "ENCODER_SETTINGS_SHA256": sha256(settings_path),
        "ENCODER_CHECKPOINT": str(checkpoint), "ENCODER_CHECKPOINT_SHA256": checkpoint_hash,
        "SOURCE_0": str(sources[0]), "SOURCE_1": str(sources[1]), "SOURCE_2": str(sources[2]),
        "ENCODER_FIT_CUTOFF_MS": str(ENCODER_FIT_CUTOFF_MS), "AVAILABILITY_LATENCY_MS": str(LATENCY_MS),
        "HISTORY_STEPS": "4", "SPACING_TICKS": "1", "HORIZONS": "1,16,128", "FUTURE_LABEL_HORIZONS": "128",
        "TRAIN_BEGIN_MS": str(endpoint(0)), "TRAIN_END_MS": str(endpoint(299)),
        "VALIDATION_BEGIN_MS": str(endpoint(300)), "VALIDATION_END_MS": str(endpoint(599)),
    }
    plan = leaf / "prep.conf"
    new_file(plan, "# Synthetic correctness fixture only, not a market research recipe.\nLATENT_MARKET_V1 {\n" +
             "".join(f"  {key} = {value};\n" for key, value in fields.items()) + "}\n")
    receipt = {
        "kind": "synthetic-cpb-v1-cache88-fixture-v1", "record_size_bytes": CACHE88.size,
        "records_per_channel": ROWS, "layout_tests": 6, "encoder_calls_by_generator": 0,
        "model_updates": 0, "scaler_fits": 0, "market_inputs": 0,
        "runtime_checkpoint_sha256_verified": checkpoint_hash, "prep_conf_sha256": sha256(plan),
        "source_closure_sha256": sha256(closure), "input_manifest_sha256": sha256(pins),
        "expected_unique_required_endpoints": 21, "expected_native_captures": 18,
        "expected_pair_origins": 3, "expected_eligible_horizon_associations": 6,
        "expected_invalid_horizon_associations": 3,
    }
    new_file(leaf / "fixture.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"Synthetic raw capture fixture ready: {plan}")
    print("3 x600 CACHE88 records; 2 eligible origins plus 1 missing/boundary origin.")
    print("Generator executed no models, fits, builds, or market measurements.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_leaf", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true", help="Check actual CACHE88 layout without writing fixture files")
    args = parser.parse_args()
    require(sys.platform.startswith("linux") and Path("/.dockerenv").is_file() and ROOT.is_dir(),
            "Run only in the documented managed Linux container; no host fallback")
    count = layout_tests()
    if args.self_test:
        require(args.new_leaf is None, "--self-test does not accept an output leaf")
        print(f"CACHE88 layout: {count} checks passed")
        return
    require(args.new_leaf is not None and args.new_leaf.is_absolute(), "An absolute new output leaf is required")
    leaf = args.new_leaf.resolve()
    require(leaf != BUILD_ROOT and BUILD_ROOT in leaf.parents, "Fixture leaf must be beneath /opt/cuwacunu/build")
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.mkdir()  # exclusive; an existing leaf is never reused or removed
    make_fixture(leaf)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Synthetic capture fixture failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
