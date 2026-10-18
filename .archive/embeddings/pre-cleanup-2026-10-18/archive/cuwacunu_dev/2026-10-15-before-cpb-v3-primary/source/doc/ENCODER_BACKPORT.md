# Current operational authority — 15 October 2026

The user explicitly selected **CPB-v1.market-01** as the sole active encoder in
cuwacunu_dev. Read [CAUSAL_PRIMARY_INTEGRATION.md](CAUSAL_PRIMARY_INTEGRATION.md)
and [the live README](../README.md) for current settings, runtime artifact,
explicit-cutoff API/CLI and verification status. This application selection
supersedes the RPB-v22-primary/default and live-source-path directions in the
historical text below. It does not declare an overall research winner or
resume stopped research.

The complete causal implementation remains
code/encoders/causal_patch_bottleneck_mae/versions/v1/; application inference
uses genuine F64 values, actual timestamps, masks and explicit cutoffs with the
saved frozen scaler/normalizer. No new training or quality comparison follows.

Retired dev encoder/research source is preserved byte-exactly in the sibling
[archive](../../cuwacunu_embedding/archive/cuwacunu_dev/2026-10-15-before-causal-primary/README.md);
the [manifest](../../cuwacunu_embedding/archive/cuwacunu_dev/2026-10-15-before-causal-primary/manifest.json)
maps its source/ paths to the former dev paths. Historical results,
checkpoints, ledgers and source capsules stay at their original locations.
Keep every research reference and candidate identity. Reusable causal changes
must return to cuwacunu_embedding; market adapters/runtime selection stay here.
Each future causal version owns its complete independent implementation.

The following original text is retained as historical evidence. Its
milestone-local defaults, paths and proposed next actions are not current
operational instructions.

---

# Encoder changes must return to cuwacunu_embedding

## RPB source-layout cleanup — 15 October 2026

Current source ownership and entries are in the existing
[RPB version index](../code/encoders/raw_patch_bottleneck_mae/versions/README.md).
Existing version-specific files now live under `versions/vN/`; common files
live under `shared/`. Original public include roots forward to those files,
and selected configuration paths retain their exact bytes. All 99 inherited
portable pairs still match their original hashes at their mapped paths.
The [layout audit](rpb_layout_cleanup/README.md) records every move and hash,
sharing, preservation and bounded container verification. Older receipts and
source capsules below retain their original paths and declarations. This
cleanup changes no encoder behavior, causal source or reference selection.
The requested starting commits were pushed; the cleanup remains uncommitted.

## CPB-v1 independent causal family — 15 October 2026

The requested new `causal_patch_bottleneck_mae` family is mirrored locally with
its complete physical `versions/v1/` source/config/tests/provenance, public
version entry and standalone Make fragment. No encoder implementation includes
or forwards into RPB or another causal version. New parser utilities are owned
copies too. The compiler's42-header closure proves this boundary; Torch/STL are
external dependencies. The integration build/proof is shared evidence for the
byte-identical source mirror; no second container/model run is claimed.

Read [the results](CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md) and
[the portable guide](CAUSAL_PATCH_BOTTLENECK_MAE_V1.md). Exact V30 tensors are
imported into one new typed CPB checkpoint with zero encoder updates. Cutoff
and temporal-causality proof pass, market9/9+strict8/8 but neural-trend2/3,
TEMPO-5 exact retention3/8 at full coverage. Preserve all reference selections,
55 older checkpoint pins and both engineering attempts. No commit/push or
automatic next training follows. Historical RPB indexes are metadata additions,
not moves or rewrites of old implementation/results. Future causal versions
own complete separate copies. Market loading and both fixed-head adapters stay
integration-owned; the exact new artifact paths/hashes live in the instance
record and result receipts.

## RPB-v30 Gaussian waveform location/scale decoder — 15 October 2026

All99 reusable pairs match locally:91 preserved plus8 new V30 header/facade/
settings/fixture pairs. The [final sync receipt](results/market_location_scale_decoder_v1/portable-sync-final.json)
pins every path/size/hash. Application raw loading, producer, orchestration and
fixed heads remain integration-owned. Only the existing integration container
executed engineering and quality; no embedding-side fixture/quality is claimed.

The optional `rpb_location_scale` module keeps exact parent32 plus learned4,
the unmodulated waveform mean and unchanged V28 relative bank/standardizer.
New variance matrix108/bias27 adds135 parameters: total297560/updated15039/
protected282521. A fixed anchored softplus link gives actual initial variance1;
proper CUDA F64 Gaussian NLL retains valid negative values. Typed inference/TRAIN
archives bind the new likelihood/link, input scaler/normalizer/source, model,
optimizer mappings and counters. Protected parent state remains exact.

The [verified market result](MARKET_LOCATION_SCALE_DECODER_RESULTS.md) passes9/9
original,8/8 strict,3/3 positive neural trend fits and full coverage. V30 leads
all four linear heads and neural volatility in the recorded matched scope;
preserve v25 neural trend/relation, native40 neural activity and all displaced
v26/v29 records. RPB-v22 stays primary and native36 stays balanced pending the
[separate frozen TEMPO-5 transfer](MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_CARD.md),
which is declared but unmeasured under both fixed scaler policies/zero updates.

Fresh saved v24 weights give exact saved-v28 full initial36 parity and zero
initial head refits. Only final one ridge pipeline/four targets and three neural
fits are new. V28 calibration/raw audit is reused with zero new fits/forwards.
One2048-update/B8 CUDA trajectory took131.75605979s including checkpoint/witness/
loader-parity I/O. Mean query MAE remains separate and does not improve uniformly;
negative NLL is legitimate. This is no fair speed benchmark or causal isolation.

Both engineering compiles passed; the first formatting warning was corrected
before the single passed GPU fixture. Masking, actual CUDA learning, exact
parent32 and typed next-update resume pass. Saved arithmetic passes941,513 checks.
Report verification binds198 assets,570 durable members and96 rows:88 preserved,
4 reused initial and4 new final. All55 checkpoint pins are available/local/ignored.
The [portable guide](../../cuwacunu_embedding/doc/LOCATION_SCALE_DECODER_V30.md)
and [exact registry](../../cuwacunu_embedding/doc/location_scale_decoder_embedding_versions.json)
pin every new checkpoint, recipe, reuse and evidence identity. Full decoder/loss
witnesses and weights stay ignored local; compact metadata/source capsules are
durable. No primary promotion, commit, push or publication.

## RPB-v29 parent-modulated waveform decoder — 15 October 2026

All 91 reusable pairs match locally: 83 preserved plus 8 new header/facade/
settings/fixture pairs. The [sync receipt](results/market_parent_modulated_decoder_v1/portable-sync.json)
pins exact paths/sizes/hashes. Application raw loading, shared-bank orchestration,
producer, run wrapper and fixed heads remain integration-owned. Only the
integration container executed the fixture and one quality trajectory.

The optional `rpb_modulated` module retains v28's exact relative bank and saved
standardizer, adding a zero-start bias-free A[27,32] gain from detached parent32
to the waveform residual. Native36 stays fixed; total/updated/protected counts
are 298289/15768/282521. New V29 typed inference/TRAIN archives bind this decoder
recipe, reused bank/scaler/source identities, optimizer mappings and counters.
The [portable guide](../../cuwacunu_embedding/doc/PARENT_MODULATED_DECODER_V29.md)
and [registry](../../cuwacunu_embedding/doc/parent_modulated_decoder_embedding_versions.json)
pin the five local checkpoints, initial scoring reuse and complete evidence.

The [verified result](MARKET_PARENT_MODULATED_DECODER_RESULTS.md) passes 4/9
original and 4/8 strict conditions, with all three neural trend fits negative.
It added then-leading linear activity/relation references0.8255824303229565/
0.4539605718451353; preserve those records after V30's later gains. Keep v22
primary and native36 balanced. Stop the
exact gain mechanism without rescue or synthetic transfer.

Fresh saved v24 initialization gives exact saved-v28 full36 parity, enabling
initial fixed-head score reuse with zero refits. New final fits are one ridge
pipeline/four targets and three neural fits. Reuse the original v28 standardizer
and raw/calibration audit with zero new fits/forwards. Preserve the first failed
fixture compile and passed build, plus the failed frozen reporter and approved
additive report-only correction. No quality execution was repeated for reporting.
Actual CUDA learning, protected state/resume and 567,804 scalar audit checks pass.
Full gains/decoder/raw witnesses and all weights remain local ignored; compact
metadata and source capsules form the durable record. The separate
[v30 card](MARKET_LOCATION_SCALE_DECODER_CARD.md) is now measured as recorded
above; its frozen transfer remains pending. No primary promotion, commit, push
or publication.

## RPB-v28 raw-F64 relative change bank —15 October2026

All83 reusable pairs match locally:73 preserved plus10 new portable headers,
facade, settings and fixture pairs. The
[sync receipt](results/market_relative_changes_v1/portable-sync.json) pins exact
paths, sizes and hashes. The application raw-F64 record loader and its per-batch
historical-F32 parity checks remain integration-owned, along with the producer,
data/scaler/target orchestration and fixed heads.

The new `rpb_relative` module receives genuine raw-F64 observations, derives
the historical parent view with raw-F32-before-scaler preprocessing, and uses
one final visible mask for both. Its270-coordinate symmetric-relative adjacent
bank and frozen TRAIN standardizer are typed reusable state. Native36 retains
the exact parent32 plus learned4. The
[portable guide](../../cuwacunu_embedding/doc/RAW_RELATIVE_CHANGE_V28.md) and
[registry](../../cuwacunu_embedding/doc/raw_relative_change_embedding_versions.json)
pin API, bank/scaler identities, all five checkpoints and the measured outcome.

The [completed run](MARKET_RELATIVE_CHANGE_RESULTS.md) passes7/9 original and
4/8 strict conditions; both volatility heads fail. All three final trend neural
fits are positive with full coverage. Keep v22 primary, native36 balanced and
all task/head leaders. CUDA admission, exact protected state/resume and1,549,413
saved arithmetic checks pass. Preserve failed build01/source and final build02.
No standalone fixture or quality execution is claimed; all tensors/witnesses
remain local ignored, with compact metadata and source capsules durable.

Stop the exact v28 mechanism without rescue or synthetic transfer. The separately
declared [v29 card](MARKET_PARENT_MODULATED_DECODER_CARD.md) reuses its bank and
normalizer but adds parent-modulated residual decoding from fresh v24 initialization.
Admission/quality are pending. New calibration fits/forwards are0; initial full36
parity must precede initial score reuse, and only final heads are newly fitted.
This work remains local without primary promotion, commit, push or publication.

## RPB-v27 parent-conditional change branch —15 October2026

All73 reusable pairs match locally:64 preserved plus9 new portable headers,
facade, settings and fixture pairs. The
[final sync receipt](results/market_parent_conditional_changes_v1/portable-sync-final.json)
pins every path, size and hash. The optional standalone Make fragment is an
environment-specific adapter rather than a claimed identical file. Market
loading, calibration orchestration, runner and fixed heads remain integration-owned.

The new `rpb_conditional` implementation adds a frozen unlabeled TRAIN map
from the exact parent32 to the existing conditioned270-coordinate bank.
Its generic map/normalizer metadata, typed inference/TRAIN state, immutable
publication and exact resume are reusable. The
[portable guide](../../cuwacunu_embedding/doc/PARENT_CONDITIONAL_CHANGE_V27.md)
and [registry](../../cuwacunu_embedding/doc/parent_conditional_change_embedding_versions.json)
record the API, geometry, map/scaler/checkpoint identities and stopped outcome.
All frozen map weights, feature/witness and optimizer archives stay ignored local
files; compact metadata and hashes are the durable record.

The [completed market run](MARKET_PARENT_CONDITIONAL_CHANGE_RESULTS.md) passes
5/9 original and2/8 strict conditions; all three neural trend fits are negative.
Keep v22 primary, native36 balanced and v26's neural-volatility head leader.
The failed first integration fixture assertion and passed second attempt remain
preserved. Actual CUDA learning, exact protected state/typed resume and2,716,060
saved-arithmetic checks pass. Source synchronization implies no standalone
fixture or quality execution. Stop v27 without rescue or synthetic transfer.
The next raw-F64 symmetric-relative bank has a
[separate v28 card](MARKET_RELATIVE_CHANGE_CARD.md), pending admission/calibration/quality.
This work remains local; no primary promotion, commit or push is implied.

## RPB-v26 TRAIN-conditioned change bank —15 October2026

The preserved [conditioning result](MARKET_CONDITIONED_CHANGE_RESULTS.md)
passes8/9 original and5/8 strict conditions. Its frozen TRAIN coordinate
mean/population-SD normalizer is reusable typed state; v27 reuses the exact
archive without refitting. V26's neural-volatility R²0.41926036121396154 remains
a task/head leader; its full market gate fails. Its
[portable guide](../../cuwacunu_embedding/doc/TRAIN_CONDITIONED_CHANGE_V26.md)
and original checkpoints remain intact.

## RPB-v25 balanced change-energy objective —15 October2026

All54 portable pairs match locally:45 historical pairs unchanged plus9 new
header/facade/settings/fixture pairs. The [sync receipt](results/market_change_energy_v1/portable-sync-final.json)
pins each path, byte size and hash. New portable code is under `rpb_energy` here
and the standalone raw-patch module there. Its optional standalone Makefile
is integration-independent; only the integration CUDA fixture was executed.
Market ingestion, the experiment runner and fixed heads stay here.

The new typed model delegates unchanged v24 native36 encode/decode and adds a
training-only4-to270 moment decoder. Separate RMS normalization, balanced
signed/energy loss and complete typed inference/resume are portable. The
[standalone guide](../../cuwacunu_embedding/doc/CHANGE_ENERGY_V25.md) explains
the explicit auxiliary-loss call and protected-parent contract.

The [completed market run](MARKET_CHANGE_ENERGY_RESULTS.md) improves trend,
activity and relation over native36 but worsens both volatility heads. Original7/9
and strict6/8 pass, so this exact objective stops without rescue or TEMPO-5
evaluation. Keep v22 primary and every reference checkpoint. New reusable
changes and results remain local, uncommitted and unpublished.

## RPB-v24 native36 variation — 14 October 2026

The separate native36 settings and CUDA fixture are locally mirrored; all
45 portable pairs match byte-for-byte, including all 43 preserved pairs.
Portable encoder arithmetic and the original native40 settings/fixture remain
unchanged. The new [portable guide](../../cuwacunu_embedding/doc/SEPARATED_CHANGE_NATIVE36.md)
and [instance registry](../../cuwacunu_embedding/doc/separated_change_native36_embedding_versions.json)
record level32 plus learned4, typed checkpoints and the completed
[market result](MARKET_SEPARATE_CHANGE_NATIVE36_RESULTS.md). Market data,
producer and readouts remain integration-owned.

One integration CUDA fixture passes exact prefix, masking/gap safety, actual
branch learning and typed resume; copied source implies no standalone run.
Native36 improves trend and passes 8/9 conditions, failing linear volatility
retention. Keep RPB-v22 primary and all v22/v23/native40/native36 evidence.
Next propose a saved-TRAIN-only covariance and fitted-linear-contribution
diagnosis, with no model updates or head refits. See
[continuation](CONTINUATION.md) and [sync receipt](results/separate_change_native36_v1/portable-sync.json).
The later [save milestone](MILESTONE_2026-10-14_NATIVE36_TRANSFER.md) records
the user's explicit commit/push authorization for this completed payload.
Checkpoint files remain local and ignored.

## RPB-v24 separate change block — 14 October2026

Six separated_change_*.h portable headers, settings and focused fixture are
locally mirrored byte-for-byte. Its [verification](results/separate_change_v1/verification.json)
checks all eight pairs plus the25 older,8 original-v23 and2 frozen-v23 pairs.
The CUDA engineering gate passes in integration; no second standalone
execution or fresh synthetic quality is claimed. The separate native40
architecture preserves the v22 native32 prefix exactly but misses balanced
market retention. Keep v22 primary and all older source/artifacts intact.
New work remains private/local; this backport authorizes no public publication.

## RPB-v23 frozen-level variation — 14 October2026

The additive consecutive_change_frozen_level.h and frozen_level_test.cpp
match the research repository byte-for-byte. Separate check targets and
training guides are present; the [receipt](results/frozen_change_v1/verification.json)
verifies both new pairs and all25 older/8 original-v23 pairs. The focused
integration CUDA gate passes; copied fixtures do not imply another execution.
No encoder inference arithmetic, primary configuration or previous artifact
changed. This is a private/local backport; no public publication is authorized.

## RPB-v23 consecutive-change candidate — 14 October 2026

Six new portable `consecutive_change_*.h` headers, the focused CUDA fixture and
settings are locally synchronized byte-for-byte. The
[verification receipt](results/consecutive_change_v1/verification.json) pins
all eight pairs and verifies the unchanged25 existing pairs. Separate optional
Make targets are available in both repositories; market loading/runner/readouts
remain in integration. The [experiment](MARKET_CONSECUTIVE_CHANGE_RESULTS.md)
improves linear trend but misses the full development preference. Primary
RPB-v22 and all original instances stay preserved. These additions remain
private/local; synchronization grants no public publication or MIT release.

`cuwacunu` integrates data ingestion with the selected encoder.
The primary selection is now RPB-v22; see
[the selection record](PRIMARY_ENCODER_SELECTION.md). Its portable architecture
already matches the standalone implementation. This selection changes the
application entry point, settings and checks, not model arithmetic.
`C:\Work\cuwacunu\cuwacunu_embedding` remains the standalone representation
research repository. The initial import left it unchanged; the user subsequently
authorized reusable model and core settings changes in both repositories.

Any reusable encoder improvement developed here must be brought back there
as a separate, reviewable local change. New original work is reserved under
the root [LICENSE](../LICENSE). Synchronization does not authorize an MIT grant
or a push to the public standalone repository; those require an explicit user
decision. Preserve existing MIT portions and identify private additions
separately, following [the retained notices](../LICENSES/NOTICE).
Examples:

- Configurable input axes, channel/feature counts, history length, patching or
  output embedding width; explicit distinction between tensor rank and axis size.
- Mask propagation, variable-length handling, encoder blocks, reconstruction
  heads, objectives, serialization and model-only performance improvements.
- Scaler fitting improvements needed for batches or streaming datasets.

The market dataloader, file/cache formats, source selection, timestamp alignment
and application adapters stay in `cuwacunu`. They are not required dependencies
of the standalone encoder and should not be backported as such.

For each reusable change:

1. Record the encoder design tag, affected contract and original source revision.
2. Implement it here without altering imported v19 weights or historical evidence.
3. Add focused shape/mask/checkpoint checks. Changes to model arithmetic or geometry
   require a new design version; do not relabel v19 or overwrite its checkpoints.
4. Port only the model/scaler changes and relevant checks into the encoder-owned
   and shared model directories of `cuwacunu_embedding`. Keep data dependencies out.
5. Record the backport commit here once complete. Until then, list it as pending.

Implemented: **RPB-v20**, separately tagged configurable spectral encoder.
Its portable config, relation bank, model, typed file binding and inference
checkpoint headers are copied byte-identically into the standalone encoder.
The core parser/owning file module is copied into its shared include tree;
model/config/checkpoint tests and default settings are copied too. Standalone
targets are `rpb-v20` and `test-rpb-v20`; old CLI/defaults and protocol drivers
keep their original behavior.

The [backport manifest](CONFIGURABLE_ENCODER_BACKPORT.json) records all 15
file pairs and SHA-256 values. Original imported headers remain unchanged;
the exact v20 settings now live at `config/encoder_v20.conf`, while
`config/encoder.conf` selects v22. Only the live settings-pair path is updated;
its original SHA-256 and all historical evidence remain unchanged.
v19 stays `[B,3,32,3] -> [B,32]`. The new native contract is `[B,C,H,F] -> [B,D]`,
with all nine kline features in the integration adapter. No market dataloader or
application adapter is backported. The standalone repository now retains a
completed synthetic TEMPO-5 C3/H128/F9 capacity experiment. This does not
establish market quality or quality for every supported geometry.

Validation and repository revisions are recorded in
[CONFIGURABLE_ENCODER_VALIDATION.md](CONFIGURABLE_ENCODER_VALIDATION.md).

Initial standalone backport commit: `37c7a3be67a359f532aad238e54639ce9913bf39`
(`Add configurable RPB-v20 encoder and reusable core settings`). No reusable
encoder change remained pending for that milestone. Streaming scaler fitting
was then the next separate candidate; the additive module is now implemented
as recorded below. The subsequent
[stability review](STABILITY_REVIEW_2026-10-12.md) also backports strict config
comment handling, model-state validation, atomic immutable checkpoint publication
and the checkpoint integrity fixture. Model arithmetic and the v20 tag remain
unchanged. The manifest records the latest completed backport revision.

The [standalone code review](../../cuwacunu_embedding/doc/CODE_REVIEW_2026-10-12.md)
additionally fixes isolated temporary checkpoint fixtures in both repositories.
Focused CUDA checkpoint checks pass here and in the standalone container; all
15 file-pair hashes match the manifest. This changes test isolation, with no
model arithmetic or configuration change. Evaluation-specific archive/admission
and saved-result audit fixes stay in the standalone repository. Its original
quality measurements and all 815 artifact hashes pass the stronger saved-only
review unchanged. Completed backport commit:
`9a1e614584036baad790593223d7174cec1c3b90`.

The completed [saved-coordinate diagnostic](../../cuwacunu_embedding/doc/CONFIGURABLE_BRANCH_V1.md)
fits only diagnostic heads to existing standalone arrays; it changes and executes
no encoder. Complete32 quality scores and checkpoints remain unchanged. Its
[synchronization receipt](../../cuwacunu_embedding/doc/results/configurable_branch_v1/completed/encoder-sync.json)
verifies all15 shared file pairs against this manifest. No additional model
backport is pending. Evaluation additions remain in the standalone repository;
its [latest continuation](../../cuwacunu_embedding/doc/CONTINUATION_2026-10-12_AFTER_CONFIGURABLE_BRANCH.md)
describes the next learned-path investigation.

## Streaming and resume support — 13 October 2026

The additive `StreamingScaler` header/test and distinct v22 training-state
header/test are mirrored byte-for-byte into `cuwacunu_embedding`; focused Make
targets exist in both repositories. Actual container tests pass for masked,
semantic-coordinate population moments and exact next CUDA update after AdamW
reload, including an ineligible attempt, gradients, parameters and moments.
The [portable support guide](STREAMING_TRAINING_SUPPORT.md) records APIs, four
pair hashes, binaries, logs and the repaired archive-reader aliasing bug.
All15 original v20 and all6 v22 portable pairs were rechecked unchanged.
No model architecture, original checkpoint or historical quality evidence changes.

The bounded market runner, Binance acquisition, chronological selections and
four all-past target/readout definitions stay in this integration repository.
Evaluation consumes saved features and targets and has no encoder dependency.
These application files are not backported with the portable headers. Local
synchronization does not authorize publication of private additions to the
public embedding remote. The admitted market pilot is recorded separately
with its own trained-instance tag.

Completed local backport commit:
`f3fef0bf9c4ef795cf31d7c0334bcb8d1d72a4e6`
(`Backport streaming scaler and resumable v22 training support`). The integration
implementation/results commit is `63df64b`. Both are local saves; no new private
work was published to the embedding remote. All four new code/test pairs and
the portable support guide remain byte-identical after committing.
RPB-v26 local stage,15 October2026: ten new reusable headers/facade/config/fixture
pairs plus the prior54 match byte-for-byte. The portable context-specific
`conditioned_change.mk` adapter is separate. CUDA admission and one market run
are verified; original8/9 and strict5/8 fail the complete gate. Preserve the
new neural-volatility task/head leader, every checkpoint and native36 balanced
reference. See `MARKET_CONDITIONED_CHANGE_RESULTS.md`; no later publication is
authorized. The separately declared RPB-v27 parent-conditional map is pending.
