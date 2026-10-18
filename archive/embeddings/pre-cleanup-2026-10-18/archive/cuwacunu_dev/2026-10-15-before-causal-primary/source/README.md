# Cuwacunu

For encoder research continuation, read the [14 October session handoff](doc/HANDOFF_2026-10-14.md).

A clean integration of the selected content-only relation encoder, the memory-mapped market
dataloader and core file/settings parser extracted from `cuwacunu_torch`.

The local development checkout is `C:\Work\cuwacunu\cuwacunu_dev`.
The project and repository name remain **cuwacunu**. This integration repository
uses the private [savethebeesandseeds/cuwacunu](https://github.com/savethebeesandseeds/cuwacunu)
GitHub remote. Keep it private until the user explicitly authorizes public
publication. The older book project is separate at
[cuwacunu_book](https://github.com/savethebeesandseeds/cuwacunu_book).

New original code and documentation are **all rights reserved** while the
open-source decision is pending; see [LICENSE](LICENSE). Existing MIT
components retain their notices and permissions, recorded in
[LICENSES/NOTICE](LICENSES/NOTICE) and [LICENSES/MIT.txt](LICENSES/MIT.txt).

**RPB-v22** is the primary configurable encoder. **RPB-v20** remains its reusable
base and an explicit reference. **RPB-v19** and its confirmed checkpoints
remain the frozen reference. Reusable encoder code is also present in
`cuwacunu_embedding`; the historical experiment collection stays there.

The [RPB-v22 content-only relation API](code/encoders/rpb_content/README.md)
reuses the v20 base, scaler and past-only input adapter. Its synthetic
development gates and initial unified CUDA check pass. Use
`cuwacunu::encoder::primary` and `config/encoder.conf` for normal application
integration. Exact v20 settings remain in `config/encoder_v20.conf`, and the
explicit `content` API and `config/encoder_v22.conf` remain available.
See [the primary selection record](doc/PRIMARY_ENCODER_SELECTION.md) and the
historical [initial sync receipt](doc/CONTENT_ENCODER_SYNC.json).

The first [market-history pilot](doc/MARKET_HISTORY_PILOT_RESULTS.md) is complete:
streaming TRAIN statistics, resumable GPU training and encoder-independent
fixed readouts on public BTC/ETH/SOL minute candles. **RPB-v22.market-01**
improves volatility/activity representation; past trend remains weak. This is
historical representation evidence, with no forecasting evaluation. Follow
[the runner instructions](code/training/README.md); its market settings are
separate from the primary API's default settings.

The additive [training/dimension study](doc/MARKET_TRAINING_STUDY_RESULTS.md)
extends a separate copy to 2048 updates and compares internal width, native size
and longer history on a shared recent-history panel. Its
[data contract review](doc/DATA_CONTRACT_REVIEW_2026-10-14.md) checks exact
timestamps, role boundaries, masks and H128/H256 suffix parity. The original
pilot and all synthetic references remain preserved; the study selects no
market-serving checkpoint.

The separately tagged [RPB-v23 consecutive-change candidate](doc/MARKET_CONSECUTIVE_CHANGE_RESULTS.md)
is implemented and locally mirrored to the embedding repository. Its matched
GPU experiment improves linear trend and relation, with a volatility tradeoff
and still-negative neural trend. It remains an explicit candidate; RPB-v22
stays primary and all saved instances remain preserved.

```text
cuwacunu_dev/
├── code/
│   ├── data/                 CSV/cache/mmap, causal windows and batching
│   ├── config/               Core file parser and typed settings
│   ├── encoders/rpb_spectral/ Reusable RPB-v20 base and reference
│   ├── encoders/rpb_content/  Primary RPB-v22, sharing the v20 base
│   ├── encoders/rpb_changes/  Optional RPB-v23 consecutive-change candidate
│   ├── encoders/rpb_v19/     Frozen selected reference and internal headers
│   ├── integration/          Explicit past-only data-to-encoder adapter
│   ├── training/             Bounded mmap/GPU pilot with strict resume identity
│   ├── evaluation/           Independent saved-feature regression readouts
│   └── scripts/              Container tasks and reference checks
├── doc/                      Contracts, provenance and continuation
├── config/encoder.conf       Primary RPB-v22 geometry, metadata and CUDA settings
├── config/encoder_v20.conf   Exact preserved RPB-v20 reference settings
├── config/encoder_v22.conf   Explicit standalone-compatible RPB-v22 settings
├── output/reference/         Local confirmed weights and parity fixtures (ignored)
├── Makefile
├── setup.sh                  Dependency setup and installed-environment reuse
├── container.ps1             Managed GPU development container
└── environment-image.json    Pinned local copy of the installed environment
```

## What is available

| Component | Available |
| --- | --- |
| Encoder | Primary RPB-v22: `[B,C,H,F] -> [B,D]`; default geometry C3/H128/F9/D32 |
| Reference | RPB-v19, unchanged `[B,3,32,3] -> [B,32]` |
| Trained references | All six `RPB-v19.alt-01` confirmed instances, copied with SHA-256 records |
| Dataloader | On-demand mmap windows, nine kline features, masks, timestamps and configurable batch/history sizes |
| Integration | All nine features, row/per-value masks, common regular time grids; future values excluded |
| Settings | Configurable channels, history, feature count, embedding size and model settings; core parser only |
| Training | Streaming TRAIN scaler; distinct v22 model/scaler/AdamW/counter checkpoints; GPU runner |
| Evaluation | Independent fixed readouts for four past-history properties; bounded MKT-HIST-1 pilot complete |
| Environment | Dedicated GPU container copied from the installed embedding environment; loopback port 4873 reserved |

**RPB-v19** combines 20 learned shape coordinates with 12 fixed spectral
relation coordinates. Its unchanged-design confirmation is **RPB-v19.alt-01**.
The strong synthetic results are reused evidence, not new market results.
See [the preserved confirmation results](doc/spectral_confirmation_v1.json)
and [encoder provenance](doc/IMPORTS.md).

**RPB-v20** generalizes that allocation: four fixed coordinates per channel
pair, with the rest of the requested embedding learned. Its reference retains
20 learned + 12 fixed. Actual CUDA checks pass for the generalized model;
market quality remains unmeasured. No original trained instance was changed.

**RPB-v22** adds learned normalized time-delay relationships to those learned
coordinates, while preserving the fixed spectral tail. Choosing this architecture
does not select a trained market checkpoint. Its synthetic research checkpoints
and historical reference weights remain separate.

Edit [config/encoder.conf](config/encoder.conf) to set `HISTORY_LENGTH` and
`EMBEDDING_SIZE`. Use that same history setting when constructing each loader
source. H must be at least 17; it need not divide evenly into patches. C/H/F/D
are fixed per model/checkpoint, while batch size can vary. Every channel must
share the configured time grid. Different candle intervals require a declared
alignment procedure. Nine-feature geometry needs newly trained weights and a
TRAIN-fitted scaler; preserved v19 checkpoints still accept only their old shape.

## Build and check

Follow [the environment instructions](doc/ENVIRONMENT.md). From PowerShell:

```powershell
.\container.ps1 up
.\container.ps1 -Action exec -Command @('bash', 'setup.sh', '--no-download')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'import', '-j2', 'test')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'primary-v22', '-j2', 'test-primary')
.\container.ps1 -Action exec -Command @('bash', 'code/scripts/task.sh', 'geometry', '-j2', 'test-v20')
```

Build products and temporary files stay inside the Linux container. Tests cover
the extracted storage behavior, actual CUDA encoding, fixed-prior/checkpoint
preservation, the data adapter, and saved-reference inference parity.
Reference checks need the local ignored artifacts listed in
[REFERENCE_ARTIFACTS.json](doc/REFERENCE_ARTIFACTS.json); they are not Git-tracked.

Public APIs: [dataloader](code/data/README.md), [core settings](code/config/README.md),
[primary encoder](code/encoders/rpb_content/README.md),
[v20 base/reference](code/encoders/rpb_spectral/README.md) and
[frozen reference](code/encoders/rpb_v19/README.md).

Reusable encoder improvements must return to `cuwacunu_embedding`, as described
in [ENCODER_BACKPORT.md](doc/ENCODER_BACKPORT.md). The market dataloader stays here.
See [CONTINUATION.md](doc/CONTINUATION.md) before starting market training.
The [in-depth stability review](doc/STABILITY_REVIEW_2026-10-12.md) records
repaired data/checkpoint edge cases, verification and the remaining pilot scope.

The separate [RPB-v23.market-01.alt-01 experiment](doc/MARKET_FROZEN_CHANGE_RESULTS.md)
protects a trained v22 child while learning only the change projection.
Its CUDA protection/resume checks pass, but trend and two linear retention
conditions fail. Keep v22 primary and all instances intact. The additive helper
is locally synchronized; the [continuation](doc/CONTINUATION.md) proposes
separate level/change coordinates as the next investigation.

The explicit [RPB-v24 separate change block](doc/MARKET_SEPARATE_CHANGE_RESULTS.md)
preserves the trained v22 output32 and appends8 learned change coordinates.
Its GPU experiment reaches0.599 linear trend R² and positive0.066 neural mean,
with volatility/relation tradeoffs. It is preserved alongside v22/v23, with
v22 still primary. The [module guide](code/encoders/rpb_separate/README.md)
documents configurable level/native dimensions and typed checkpoints.
