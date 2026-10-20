# V16 temporal study: March confirmation data owner

This is a new path-bound source generation copied from
`../cpb_v15_objective_v1/`; [source-copy-provenance.json](source-copy-provenance.json)
binds the original and new source bytes. It changes no V15 source or evidence.
The copied CACHE88 parser retains exactly the V15 bytes, including its historical
`mityane::data::cpb_v15` namespace. That namespace is source provenance, not a V15
dataset, experiment, receipt or authority claim.

Only BTCUSDT, ETHUSDT and SOLUSDT Spot1m monthly klines for March2024 are admitted,
in that order: 44640 rows, open `1709251200000`, first close `1709251259999`, last
close `1711929599999`. No substitution, unit coercion, row replacement, filtering,
normalization, gap filling or feature/target calculation occurs here. The ZIP
member, publisher checksum, twelve-column CSV, finite values and complete
millisecond lattice must all match. Market files and generated artifacts remain
ignored under `.temp/data/cpb-v16-temporal-001/`.

No source editing declares or executes a study. The root-owned V16 protocol,
actual source/build/admission reviews and explicit invocation are required.
Build/test only through the root Makefile/task launcher in the existing managed
container. `converter.mk` exposes `build-cpb-v16-data` and `test-cpb-v16-data`;
its products live in `$(BUILD_DIR)/cpb-v16-data`. No host toolchain is used.

## Exact preparation interfaces

`prepare.mjs metadata --declaration FILE --source-review FILE --output NEWLEAF`
fetches only the exact official README and dataset terms. Its output leaf is
`.temp/data/cpb-v16-temporal-001/publisher-metadata/attempt-NNN`. It neither
downloads market values nor invokes the converter, models, targets or heads.

`prepare.mjs import --declaration FILE --source-review FILE --build-receipt FILE
--target-parity FILE --publisher-review FILE --output NEWLEAF` admits only a new
`.temp/data/cpb-v16-temporal-001/attempt-NNN` leaf. It preserves archives,
checksums, verbatim CSV, CACHE88, converter logs and a final receipt or failure.
Existing and partial leaves are never overwritten or deleted.

The declaration has kind `cpb_v16_temporal_confirmation_data_selection_v1`,
status `prospective_selection_before_download_or_numeric_access`, exact exported
`GEOMETRY`, ordered official ZIP/checksum URLs, exported `METADATA`, and a full
current `resumption` pin `{path,bytes,sha256}`. `confirmation` requires role
`confirmation_only`, all training/head/normalizer fitting flags false,
`probes:32`, `history_legal:256`, `encoder_history:128`. Calendar row selection and
the six-run outcome barrier are enforced by the root-owned protocol, target
preparer and runner, rather than by this raw-data importer.

The passed independent source review kind is
`cpb_v16_temporal_independent_source_review_v1`; its `files` must bind all eight
exported `SOURCES` (seven owned files, including provenance, and shared Node
artifact helpers). The build receipt kind is
`cpb_v16_temporal_data_converter_build_v1`, binding that review, the native
converter and its `cache88.h`/`csv_to_cache88.cpp` sources. The converter uses
`--input CSV --output NEWCACHE --first-open 1709251200000 --rows 44640` and emits
`cpb_v16_temporal_cache88_conversion_v1`.

The root-owned January precision receipt must be passed kind
`cpb_v16_temporal_target_precision_parity_v1`: 112/32 rows,
`target_precision:'legacy_f32_then_f64'`, exact original targets/masks/IDs,
`March_values_opened:false`, zero encoder/head calls, and full pins for source,
reviews, binary, numeric result and log. Its numeric result kind is
`cpb_v16_temporal_target_precision_parity_numeric_v1`, with the same scientific
flags. Direct F64 label arithmetic cannot silently replace the historical
F32-loader then F64 target recipe.

## Publisher boundary

The two metadata URLs are the publisher's
[README](https://raw.githubusercontent.com/binance/binance-public-data/master/README.md)
and [Dataset Terms](https://raw.githubusercontent.com/binance/binance-public-data/master/TERMS_AND_CONDITIONS.md).
Read-only inspection on19October2026 found terms version1.0, updated26August2026,
and CC BY-NC-SA4.0 plus the Dataset Terms. This observation is not a fresh saved
metadata receipt or legal review. The actual metadata fetch and independent
`cpb_v16_temporal_publisher_metadata_review_v1` must bind the new declaration,
downloaded metadata and exact declared license before numeric import. A changed
term header/license requires an explicit new reviewed source/declaration; it
does not authorize accepting stale terms or silently relaxing this gate.

The declaration/review retain attribution, local personal noncommercial
non-production research, no commercial authorization, and no external dataset or
derived-artifact publication. The publisher README records microsecond Spot
timestamps from2025 onward; this March2024 adapter deliberately requires
milliseconds and rejects mixed or silently converted units.

The artificial tests cover supported ZIP encodings, names/paths/member count,
checksums/CRC, expansion limits, links/encryption, declaration/calendar/terms
changes, February relabeling, wrong chronology/count/lattice and microsecond
inputs. Their source exists here; a passing execution is claimed only by a
separate actual managed-container test log.
