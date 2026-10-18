# Encoder research continuation

## Latest completed task: CPB-v1 causal conversion, 15 October 2026

The user stopped the earlier continuous search, then requested an independently
owned strictly causal V30-derived encoder in both repositories. That task is
complete; [the report](CAUSAL_PATCH_BOTTLENECK_MAE_V1_RESULTS.md) contains four
market task panels, four TEMPO-5 panels, every repetition, physical errors,
conditional intervals, exact checkpoint/source bindings and costs.

**CPB-v1.market-01** is native36 and imports exact V30 tensors with zero new
encoder training. Its physical [version directory](../code/encoders/causal_patch_bottleneck_mae/versions/v1/)
owns all implementation/config/tests. The [family entry](../code/encoders/causal_patch_bottleneck_mae/README.md)
and [portable guide](CAUSAL_PATCH_BOTTLENECK_MAE_V1.md) explain version ownership,
the cutoff API, typed artifacts and which local files are necessary evidence.
Future versions require complete separate implementation copies.

The CUDA proof passes all128 cutoffs,2350 future value/mask cases and30 internal
temporal cases, exact transferred/frozen state and typed roundtrip. Unchanged
fixed heads pass market9/9 and strict8/8, but only2/3 neural trend repetitions
are positive. TEMPO-5 has strong absolute accuracy and full coverage, but3/8
exact native36 retention floors pass. Keep RPB-v22 primary, native36 balanced,
the old V30 market result, all task leaders and every historical result/checkpoint.
Old V30 transfer remains extraction-only/unscored. The old market loader ended
windows at their anchor; these are retention probes, not forecast or measured
leakage estimates.

Next direction is a separately declared causal-training experiment preserving
this frozen baseline and both datasets' guards. Improve neural-trend robustness
and restore hard-signal retention without tuning heads. No next training,
publication, cleanup or primary promotion is started by this completed task.
The sections below preserve the earlier search history and its original scope.

## Historical search goal,15 October2026

The user explicitly requested continuing as a goal until an encoder passes9/9,
while remembering the best encoders already found. The mirrored
[search ledger](ENCODER_SEARCH_LEDGER.md) and exact
[candidate index](encoder_search_ledger.json) preserve all trained references,
their original card decisions,55 checkpoint hashes and scoped task/head leaders.
Keep v22 primary, native36 as the8/9 balanced reference, and the useful native40,
v23/v25 task references and all displaced v26/v29 leaders. V30 now leads all four
linear heads and neural volatility in the recorded matched scope. Future wins update live leader pointers without
deleting/relabeling earlier records or treating different historical gates as
the same nine conditions.

The saved-TRAIN [change-scale diagnosis](MARKET_CHANGE_SCALE_DIAGNOSIS_RESULTS.md)
is complete, followed by separately declared v26 conditioning and v27 parent-
conditional experiments. V29's verified failure and V30's verified market pass
are recorded below. All eleven
candidate entries remain intact; stopped mechanisms stay stopped.

The [V30 market result](MARKET_LOCATION_SCALE_DECODER_RESULTS.md) passes9/9
original conditions,8/8 strict native36 floors, all3 positive neural trend fits
and full coverage. The separately declared
[frozen TEMPO-5 transfer](MARKET_LOCATION_SCALE_SYNTHETIC_TRANSFER_CARD.md) is
pending admission/extraction/scoring. It uses the exact saved final V30 checkpoint,
zero encoder updates and both fixed scaler policies. Keep native36 balanced
until all eight synthetic retention floors and coverage are verified. No V30
synthetic score or balanced promotion is recorded yet.

The goal permits separate prospectively declared mechanisms as evidence warrants;
stopped v25 stays stopped. Keep fixed heads/native exports and the native36
protection floors, with individual task leaders visible even if a later balanced
candidate trades against them. A full market pass precedes conditional frozen
TEMPO-5 protection; fresh chronology remains separate. Do not promote or publish
automatically. Earlier sections below retain the completed milestones.

## Latest market experiment: RPB-v30.market-01, 15 October 2026

The [Gaussian waveform location/scale experiment](MARKET_LOCATION_SCALE_DECODER_RESULTS.md)
passes the full prospective market gate. Final linear/mean-neural R² are
volatility0.9539128652969817/0.588276564672827,
trend0.7943655016246851/0.320209792300635,
activity0.8657318768925227/0.6135724853219218 and
relation0.48747626119524834/0.40683607861038357. Every task/head covers32/32 rows.
Trend neural repetitions are0.008674474092963624/0.5350811106482072/
0.4168737921607343. Preserve v25 neural trend/relation, native40 neural activity
and all displaced task/head records; there is no universal ranking.

V30 keeps the unmodulated parent-plus-change mean and adds zero-start variance
matrix108/bias27, using a fixed anchored softplus link with actual initial
variance1. CUDA F64 diagonal Gaussian NLL retains valid negative losses and
the original cell/channel/example hierarchy. Total/updated/protected counts
are297560/15039/282521. This changes objective/distribution together; it does
not establish a causal explanation for the earlier volatility failures.

One CUDA trajectory completed2048 new updates/B8 after2048 inherited updates.
Fresh saved v24 weights, exact saved-v28 initial36 parity and initial-head reuse
give zero initial refits; only final one ridge pipeline/four targets and three
neural fits are new. The unchanged V28 normalizer/raw calibration is reused,
with zero new fits or calibration forwards. GPU training took131.75605979s,
including checkpoint/witness/loader-parity I/O; feature I/O1.1259591679999998/
0.782120961s and final head fitting/scoring/I/O2.177286234s stay separate.
The historical normalizer fit2.929513154s is not a new cost. Raw-loader parity
and CPU parent preprocessing prevent a fair speed comparison.

Fixed-query mean MAE ends at0.28209757645215305 TRAIN/0.3901873007416725 known
VALIDATION versus0.27862535204206196/0.39052824676036835 initially; it does not
improve uniformly. First/last NLL is0.3399173832780379/−0.27151144758246093;
1449/2048 minibatches are legitimately negative. Keep this distinct from MAE
and classifier R². Variance includes mean error, not realized market volatility.

The integration CUDA fixture proves masking, exact protected parent, actual
learning and exact typed optimizer/weights/counters resume. Both engineering
compiles passed; a first formatting warning was corrected before the single
passed GPU invocation. All99 reusable pairs match. Saved scalar audit passes
941,513 checks; report verification binds198 assets,570 durable members and
96 task rows:88 unchanged old,4 explicitly reused initial and4 new final.
All55 checkpoint pins were rehashed and remain local/ignored. The exact final
inference SHA256 is `c4b1495c179738f9b24447d484220fa82fdd38c44e683a0ec7b2377e8545e1cb`.
See the [portable guide](../../cuwacunu_embedding/doc/LOCATION_SCALE_DECODER_V30.md)
and [registry](../../cuwacunu_embedding/doc/location_scale_decoder_embedding_versions.json).
Keep RPB-v22 primary and native36 balanced pending the frozen transfer. All
eleven measured entries and previous frozen reports/cards remain intact;
no primary promotion, commit, push or publication follows.

## Preserved v29 experiment, 15 October 2026

The [parent-modulated decoder experiment](MARKET_PARENT_MODULATED_DECODER_RESULTS.md)
is complete and stopped: original 4/9 and strict 4/8. All three final neural
trend fits are negative and coverage is 100%. Final linear/mean-neural R² are
volatility 0.6039119664332383/0.08978315114659648,
trend −0.6642949660659172/−0.4386622586930984,
activity 0.8255824303229565/0.6357176901460877 and
relation 0.4539605718451353/0.4016381483215014. Activity/relation were new linear
head leaders at that milestone; their records remain after V30's later gains.

V29 multiplies the canonical waveform residual by a bias-free gain
`1 + detached_parent32 @ A.T`, broadcast over time, then gathers semantic
channels. A[27,32] adds 864 zero-initialized parameters. Native36 and the original
relative bank/normalizer stay fixed; total/active/protected counts are
298289/15768/282521. The parent remains NoGrad with no optimizer moments/decay.
Fresh saved v24 initialization gives exact full36 parity with saved v28 initial
banks. Initial head scores are reused after parity, with zero initial refits;
only one final ridge pipeline/four targets and three final neural fits are new.
The exact v28 normalizer and original raw/calibration audit are reused with
zero new calibration fits/forwards. This is not trained-v28 continuation.

One CUDA trajectory completed 2048 new updates/B8 after 2048 inherited updates.
GPU training took 159.606196866 seconds including checkpoint/witness/loader-parity
I/O. Initial/final feature I/O took 1.494165288/1.259348142 seconds, final fixed-head
fitting/scoring/I/O 2.558239264 seconds, and saved scalar audit 1.7932028629984416
seconds. The reused normalizer's historical 2.929513154-second fit is not new cost.
Fixed-query TRAIN/VALIDATION MAE ends at 0.2657958780016218/0.38286811858415604.
These descriptive costs include raw-loader checks and CPU parent preprocessing;
they do not support a fair speed comparison with older runs.

CUDA admission proves decoder then projection/gain learning, exact parent32,
masking, typed-state rejection and exact next-update resume. The first fixture
compile failed on a const helper signature before any GPU invocation; preserve
that capsule and the corrected passed build. All 91 reusable pairs match locally.
Saved arithmetic passes 567,804 checks. Report verification checks 194 assets,
562 durable members and all 80 unchanged old plus 4 reused initial/4 new final
rows. The first frozen reporter stopped on an inherited check expecting new raw
witnesses; the approved additive v2 instead pins original v28 witnesses as the
card requires. Original source/stderr and the exact correction remain preserved.
No encoder/head/audit rerun or quality change occurred for this reporting repair.

All 50 checkpoint files were rehashed and remain local/ignored. See the
[portable guide](../../cuwacunu_embedding/doc/PARENT_MODULATED_DECODER_V29.md) and
[registry](../../cuwacunu_embedding/doc/parent_modulated_decoder_embedding_versions.json).
No v29 TEMPO-5 transfer ran after market failure. Keep v22 primary, native36
balanced and every old/new task leader. No promotion, commit, push or publication.

## Preserved v28 experiment,15 October2026

The [raw-relative experiment](MARKET_RELATIVE_CHANGE_RESULTS.md) is complete
and stopped: original7/9 and strict native36 protection4/8. All three final
neural trend fits are positive and coverage is100%; both volatility heads fail.
Final linear/mean-neural R² are volatility0.33383788288253224/−0.10588253413910875,
trend0.6848847455205371/0.29824873734480606,
activity0.7482093893948722/0.6079659622773474 and
relation0.364335324385759/0.31518332521714965. No trained-final task/head record
or balanced replacement follows. Initial neural activity0.7246054298126288 is
preserved as an initialization control, not a trained-final leader.

One CUDA trajectory completed2048 new updates/B8 after2048 inherited updates.
The genuine raw-F64 adjacent-relative bank uses deliberate increment-F32
rounding and F64 moments; its new frozen TRAIN standardizer used4096 natural
anchors with zero encoder calls. Parent32 and initial reconstruction remain
exact. Calibration2.929513154s, GPU training124.340118798s including checkpoint
I/O and loader parity, initial/final feature I/O0.933559986/0.748246063s and
fixed saved-head fitting/scoring/I/O2.598573382s remain separate. Fixed-query
MAE ends at0.27442921165909084 TRAIN/0.379517525434494 known VALIDATION.
Every raw batch is checked against the legacy F32 loader and parent preprocessing
includes CPU rounding/scaling; these descriptive timings support no fair speed claim.

Saved arithmetic passes1,549,413 checks; report verification retains all72 old
plus8 new task rows,184 asset bindings and277 durable members. The failed first
producer compile and exact source are preserved beside successful build02;
no calibration or model execution occurred in that failed compile. All83
portable pairs match; all45 candidate checkpoints were rehashed and remain
local/ignored. See the [portable guide](../../cuwacunu_embedding/doc/RAW_RELATIVE_CHANGE_V28.md)
and [registry](../../cuwacunu_embedding/doc/raw_relative_change_embedding_versions.json).
No v28 transfer ran. Keep v22 primary, native36 balanced and all partial leaders.
The next v29 declaration above changes decoder interaction, without rescuing
the stopped v28 recipe. No promotion, commit, push or publication follows.

## Preserved v27 experiment,15 October2026

The [parent-conditional experiment](MARKET_PARENT_CONDITIONAL_CHANGE_RESULTS.md)
is complete and stopped: original5/9 and strict native36 protection2/8.
All three final neural trend fits are negative; coverage is100%. Final
linear/mean-neural R² are volatility0.5237963525418771/0.19278414044391026,
trend0.11759449438517522/−0.15534257414437444,
activity0.7417729793945342/0.6210865037629124 and
relation0.3608922714799607/0.2689857305595533. No new task/head maximum or
balanced replacement follows. Keep v22 primary and native36 balanced, with
all v25/v26 and older partial leaders preserved.

The new frozen unlabeled TRAIN map removes the component of the conditioned
270-coordinate change bank predicted from parent32 before its learned4
projection. It reuses v26's exact bank standardizer; parent32 and its decoder
remain protected, with297425 total/14904 active/282521 protected parameters.
The map adds8910 frozen coefficients plus statistics, not trainable parameters.
This is approximate regularized residualization, not an orthogonality or causal
explanation of validation failure.

One CUDA trajectory completed2048 new updates/B8 after2048 inherited updates.
Parent calibration used4096 natural-mask TRAIN anchors and512 necessary frozen
CUDA forwards with zero optimizer updates; no bank-normalizer refit or held-out
calibration occurred. Parent extraction14.210417854s, map fit1.730320335s and
GPU training148.800206036s including checkpoint I/O are separate. Fixed-query
MAE is0.27492174186876844 TRAIN/0.3880198262631893 known VALIDATION, versus
initialized0.27862535204206196/0.39052824676036835. Reconstruction improves
slightly without meeting the representation gates.

The integration fixture proves masking, exact parent/initial reconstruction,
actual decoder then projection CUDA learning, absent protected optimizer state
and exact typed next-update resume. Both engineering attempts are retained;
the first rejected a fixture's overly strict cross-layout suffix equality.
The corrected numerical assertion and exact direct arithmetic witness pass
without changing model headers or producer. Saved-only arithmetic passes
2,716,060 checks; independent report verification preserves all64 old plus8
new task rows. All73 portable pairs match locally. The
[portable guide](../../cuwacunu_embedding/doc/PARENT_CONDITIONAL_CHANGE_V27.md)
and [registry](../../cuwacunu_embedding/doc/parent_conditional_change_embedding_versions.json)
pin the five new ignored checkpoints, frozen map/normalizer and evidence.

No v27 transfer ran after market failure. Stop this exact map without penalty,
normalizer, head, rate, width, seed or budget rescue. All40 candidate checkpoints
remain indexed and ignored in its frozen stage. V28's later result is above; no primary
promotion, commit, push or publication follows.

## Preserved v26 conditioning result,15 October2026

[RPB-v26.market-01](MARKET_CONDITIONED_CHANGE_RESULTS.md) passed8/9 original
and5/8 strict conditions. It retains a useful neural-volatility head record,
R²0.41926036121396154, while linear volatility, strict linear activity and
neural relation fail. Its exact uniform-conditioning mechanism remains stopped.
Keep native36 as the balanced reference and all five v26 checkpoints.

## Preserved v25 experiment,15 October2026

The [balanced change-energy experiment](MARKET_CHANGE_ENERGY_RESULTS.md)
is complete and stopped: original7/9 conditions and strict native36 protection6/8.
Both failed conditions concern volatility. Final linear/mean-neural R² are
volatility0.409235/0.236161, trend0.774228/0.488465, activity0.755266/0.597520
and relation0.373557/0.407150, all32/32 coverage. V25 is a useful stronger
trend/activity/relation reference, not a balanced replacement. Keep v22 primary
and v24 native36 as the leading balanced research reference; preserve every
v23/native40/native36/v25 checkpoint and frozen result.

One new CUDA trajectory completed2048/B8 updates after2048 inherited parent
updates, using the exact saved native36 initialization and unchanged MKT-HIST-2
data/input-scaler/targets/query plans/streams/fixed heads. Native36 and prefix32
remain exact at initialization; final prefix32 remains exact. The training-only
4-to270 decoder and balanced signed/energy loss add no exported coordinates.
Training took170.846221952 seconds including checkpoint I/O; auxiliary TRAIN-only
RMS calibration took2.192791702 seconds and zero encoder calls. Fixed-query MAE
is0.259617 TRAIN/0.388090 known VALIDATION, versus initialized0.278625/0.390528.

CUDA admission proves masking, actual learning and exact typed next-update
resume. Saved-only arithmetic passes31,851 checks; the saved TRAIN-context
reader opens13 TRAIN archives and passes1,024 scalar/tensor checks, with no new
encoder calls or fits. All54 portable pairs match locally. The
[completion](results/market_change_energy_v1/completion.json) and separate
preservation receipts retain exact sources, five new checkpoints and readouts.
All77,831 baseline files are unchanged after the run; the
[preservation check](results/market_change_energy_v1/preservation-after-summary.json)
passed. The [final verification](results/market_change_energy_v1/final-verification.json)
also binds the live handoff snapshots and checkpoint availability.

The auxiliary learned its sampled energy objective, but it did not repair
volatility generalization. Neural TRAIN volatility R² is0.929429 while known
VALIDATION is0.236161. Linear last56 TRAIN remains below the parent
(0.902583 versus0.932627); these reused fitted-head subdivisions are descriptive,
not causal proof. Read the [follow-up](MARKET_CHANGE_ENERGY_FOLLOWUP.md) before
new work: next propose a bounded saved-TRAIN check of centered, dimensionless
change statistics against the existing scaled raw-difference RMS bank. No new
diagnosis or architecture has run. Do not tune this stopped recipe or run
TEMPO-5 to rescue it. New source/docs/evidence remain local; no commit or push.
Earlier sections below retain their milestone-local proposals and dispositions.

## Latest completed diagnosis: saved TRAIN volatility, 14 October 2026

The [audited diagnosis](MARKET_TRAIN_VOLATILITY_DIAGNOSIS_RESULTS_READABLE.md) is complete:
22 hash-pinned saved TRAIN archives,112 rows, zero encoder calls/updates,
head/scaler fits or held-out tensor diagnostics. Independent arithmetic passes
15,428 checks;1,249 historical files and all nine handoff/native36 checkpoints
remain unchanged. The full per-task tables, source snapshots and receipts are
saved; native arrays and the numerical witness remain local under ignored output.

Parent32 is exact and its old predictor is still representable, but joint ridge
fitting shifts volatility prefix coefficients by L2 0.602036. TRAIN linear R²
improves0.767220→0.832289. Strong pairwise duplication is not observed
(maximum parent/change|r|0.320435), and the full volatility contributions reinforce
overall rather than cancel. Benefit is uneven: first56 TRAIN R² improves
0.654690→0.772730; last56 falls0.932627→0.910444. These are in-sample diagnostics,
not proof of the cause of worse held-out generalization or new quality evidence.

**Next proposed experiment:** retain waveform reconstruction and add balanced
reconstruction of the existing signed-change/RMS moment bank from the same four
change coordinates. Read the [next-direction proposal](MARKET_CHANGE_ENERGY_NEXT_DIRECTION.md)
for scope and strict no-decline protection of native36's other market/TEMPO-5
gains, alongside the unchanged volatility floor. This is not implemented or
quality-tested; declare its exact objective/scaling/coefficient, separate instance
and typed state before execution. Keep parent32, native36,2048/B8 and fixed heads.
Do not lead with a generic decorrelation penalty as an established remedy.
The completed diagnosis supersedes earlier proposals to perform it below.
No new reusable encoder arithmetic was introduced; new diagnosis evidence is local.

## Saved milestone and next priority, 14 October 2026

**RPB-v24.market-01.alt-01 native36 is the leading research candidate;
RPB-v22 remains the primary encoder.** The
[saved milestone](MILESTONE_2026-10-14_NATIVE36_TRANSFER.md) joins the completed
market and exact-checkpoint TEMPO-5 transfer evidence without changing either
frozen report. The user requested this work be documented, committed and pushed
to both verified existing remotes; that authorization supersedes earlier
local-only wording for this payload. Checkpoints remain local and ignored.

The next priority is the separately declared saved-TRAIN-only linear-volatility
diagnosis below. The unchanged retention floor is R² 0.6781069298378846;
native36 measured 0.44631541040772105. No diagnosis or new experiment was run
while saving this milestone. Other proposed tracks remain separate.

## Latest completed milestone: frozen market encoder on TEMPO-5

The [frozen transfer result](MARKET_SYNTHETIC_TRANSFER_RESULTS.md), declared
before quality testing, answers yes for **TEMPO-5, complexity 5/5**.
The exact saved **RPB-v24.market-01.alt-01** market checkpoint, native36,
reaches 98.96%/97.74% linear/neural accuracy intact and 93.36%/92.53% with the
saved additional 30% erasure, using the original synthetic TRAIN scalers.
These means exceed the reused synthetic-trained RPB-v22.alt-01 results in
both heads and views. The separately declared unchanged market-scaler view
also performs strongly: 100%/99.52% intact and 96.09%/95.05% deleted.
All rows have 100% coverage across three known cohorts, 512 TRAIN/256 VALIDATION
each, B8 CUDA inference and three fixed head repetitions. There were zero encoder
updates, scaler refits or old encoder/control-head replays; all model values,
91 protected files and 15 saved feature files remain unchanged.
Keep RPB-v22 primary and every historical artifact. This measures TEMPO-5
transfer only; TEMPO-4/AMP-2 have incompatible C3/H32/F3 geometry and remain
unmeasured for this checkpoint. The next priority is the saved-TRAIN volatility
diagnosis below; a new compatible hard family remains a separate future option.
Earlier sections retain their milestone-local evidence and proposals.

## Latest completed variation: RPB-v24.market-01.alt-01 native36

The [native36 result](MARKET_SEPARATE_CHANGE_NATIVE36_RESULTS.md) is complete:
the exact trained v22 prefix32 plus four learned change coordinates reaches
trend R² 0.655 linear / 0.134 mean neural, with all three neural fits positive.
Eight of nine unchanged conditions pass. Only linear volatility retention
fails: 0.446 versus the saved parent's 0.728, a loss of 0.282 R². Activity,
both relation heads and neural volatility meet their retention conditions.
Keep RPB-v22 primary; this is useful development evidence, not an overall
winner or a promotion. Preserve native40, native36 and every earlier instance.

One CUDA trajectory completed 2048 new updates/B8 after 2048 inherited parent
updates on the same MKT-HIST-2 bank/scaler/targets. Initial/final native36 banks
use unchanged fixed heads with predeclared D36 counts/seeds; all older scores
are reused. GPU training took 93.377385897 seconds including checkpoint I/O.
The integration CUDA fixture proves masks/gaps, exact prefix, real branch
learning and typed next-update resume. All 45 portable pairs match locally;
no standalone fixture execution or fresh synthetic quality is claimed.
See the [completion](results/separate_change_native36_v1/completion.json),
[preservation check](results/separate_change_native36_v1/preservation-after.json)
and [new instance registry](../../cuwacunu_embedding/doc/separated_change_native36_embedding_versions.json).

**Next proposed task:** declare a saved-TRAIN-only diagnosis of why the extra
four coordinates hurt fixed linear volatility. Inspect saved level/change
covariance and fitted linear contributions/predictions using existing banks
and head archives. Use no encoder updates, head refits or old encoder replays,
and no VALIDATION/TEST diagnostic inputs. These possible contributors are not
yet a demonstrated cause. Declare a separate new objective or branch design
only after that diagnosis; no width, rate, budget or head rescue. Fresh
chronological confirmation follows a useful balanced candidate. The diagnosis
is proposed, not measured or scheduled. The save authorization above governs
publication of the completed source/docs/evidence; weights remain local.

The sections below retain earlier milestone-local proposals. The completed
native36 variation and the next task above supersede their next-action wording.

For a new session, start with the [14 October handoff](HANDOFF_2026-10-14.md),
including exact checkpoint locations, local-only artifacts and next-task scope.

## Latest completed attempt: RPB-v24.market-01

The [separate-block result](MARKET_SEPARATE_CHANGE_RESULTS.md) preserves the
complete trained v22 native32 prefix exactly and appends8 learned change
coordinates, native40, with a separate residual decoder. One2048-update
CUDA trajectory after the parent's2048 updates, same prepared MKT-HIST-2
bank/scaler/targets and unchanged fixed-head algorithm. Both initial/final
new40 banks are evaluated; all old encoder/head results are reused.

Trend improves through learning: linear−0.721→0.599, neural−2.088→0.066.
Final linear trend exceeds the saved useful v23 score0.387. Activity is strong,
including an already strong initial neural score. Volatility/relation fail
retention against v22; the full declared goal passes5of9, so no promotion.
One neural trend fit remains negative. Head dimensions/counts/actual seeds
change with native40, while both new views share them. Keep comparisons scoped.

**Preserved reference points:** v22 remains the primary architecture and
protected level/relation baseline; RPB-v23.market-01 is the useful native32
trend reference; RPB-v24.market-01 is the stronger tested native40 trend
reference. None is declared the universal best market encoder. Keep all exact
sources, settings, checkpoints and reports; no overwriting/relabeling them.
The [v24 registry](../../cuwacunu_embedding/doc/separated_change_embedding_versions.json)
pins its new inference/training hashes, parent and initial control.

**Next proposed attempt:** a separately declared v24 variation with a smaller
change block, for example4 new coordinates/native36. Keep the same parent,
data, loss and2048 budget; declare the new dimensions/head seeds before
quality. Seek to retain useful trend while reducing added fitting inputs.
Do not tune heads, extend this stopped card or replace the saved v24 result.
A balanced candidate then needs fresh chronological confirmation. This
smaller variation is not implemented or scheduled here.

The CUDA gate passes exact prefix/reconstruction parity, masking/gaps/poison,
partial history, programmatic dimensions, actual decoder/projection learning,
protected values/no moments and exact next resumed loss/weights. Six portable
headers, settings and fixture are locally mirrored. Verification preserves
all189 capacity artifacts, both33-artifact v23 runs, primary config and
original512 checkpoint. The earlier sections record completed milestones.

## Latest completed variation: RPB-v23.market-01.alt-01

The [frozen-level experiment](MARKET_FROZEN_CHANGE_RESULTS.md) is complete.
It starts from trained RPB-v22.market-02 and adds2048 updates to only the
change projection. Exact protected child parameters/buffers, absence of child
AdamW moments/decay and typed next-update resume pass on CUDA. Initial native
features equal the saved trained v22 bank. No old encoder/head quality replay.

Four of eight declared preferences pass. Both trend heads remain negative;
linear volatility/relation lose more than.05 R². Activity and neural relation
improve versus the parent. Keep this instance and all older checkpoints intact.
Primary remains RPB-v22. The prior fresh v23 run still has the strongest trend
scores among these three trained native32 candidates; neither v23 recipe meets
the full market development goal.

**Next proposed architecture:** preserve the old32 coordinates exactly and
append a small separate change block, with a separate reconstruction path.
Declare the larger native dimension and fixed-head geometry/seed consequences
before quality. Prove old-block equality, masks/no adjacent-gap bridging and
actual CUDA learning/resume. One bounded shared-bank development test, followed
by a fresh chronological cohort if useful. This is a proposal, not implemented.
Do not increase this stopped recipe's budget or tune heads to rescue it.
No forecast, TEST or serving-checkpoint selection follows.

The additive portable freeze helper and fixture are mirrored locally in
cuwacunu_embedding. [Verification](results/frozen_change_v1/verification.json)
checks all25 older pairs,8 earlier v23 pairs and2 new pairs; all189 capacity
artifacts,33 earlier v23 artifacts and the original512 checkpoint are intact.
The source closure and exact run evidence are preserved in the new receipt.
The sections below record the earlier completed milestones and proposals.

The [RPB-v23 market experiment](MARKET_CONSECUTIVE_CHANGE_RESULTS.md) is
complete. One fresh native32 trajectory adds a zero-start observed-change bank
to the unchanged v22 child. Its initial features match the saved v22
initialization exactly; all controls and their head scores are reused.
The matched2048-update run has actual CUDA gradients/updates throughout,
independent fixed readouts and preserved typed training/inference checkpoints.

Linear trend improves−0.384→0.387, neural trend−0.542→−0.166. Relation improves
under both heads, activity stays close, and query validation MAE improves.
Volatility loses0.073 linear/0.121 neural R²; the predeclared positive-neural-
trend and maximum.05-loss preference fails. Keep RPB-v23.market-01 as useful
evidence, with RPB-v22 primary. These remain known-source history properties,
not forecasting, TEST or a universal encoder ranking.

**Next:** retain the useful change information while protecting level and
volatility information. A separately declared candidate could constrain the
change residual or separate internal allocations. The results show a tradeoff,
without proving which cause/remedy is correct. Keep native32, the same saved
MKT-HIST-2 bank and fixed heads initially. Do not change dimensions, tune heads,
repeat favorable seeds or extend the stopped card to hide that tradeoff.
Broader claims need a fresh chronological cohort after a useful candidate.

Six portable headers, the CUDA fixture and settings match byte-for-byte in
both repositories. Market data/runner/evaluation stay here. New archives are
explicitly typed v23; v22 source and archives remain preserved. The focused
engineering check passed, including masked poisoning, partial model forward,
canonical IDs, zero-energy gradients, initialized parity and exact AdamW
next-update resume. The inherited spectral base requires H>=17; the existing
coordinate-erasure helper requires complete patches, as the H128/P8 market
recipe does. No second standalone test or fresh synthetic quality was claimed.

## Completed duration and dimension context

The [14 October study](MARKET_TRAINING_STUDY_RESULTS.md) is complete. One
separate copy of RPB-v22.market-01 continued from512 to2048 updates, and four
fresh matched configurations completed2048 updates each on the GPU. All fixed
readouts consumed saved embeddings; no encoder ran again on CPU. No saved
checkpoint, primary API, architecture or synthetic result was replaced.

Extra duration alone did not solve trend: RPB-v22.market-01.alt-01 lost linear
volatility/relation performance versus the preserved512-update instance. On
the separate MKT-HIST-2 panel, RPB-v22.market-02.alt-01 (W128/D32/H128) improves
linear volatility/activity/relation versus the fresh W64 baseline, while trend
remains negative. D64 and H256 do not provide consistent benefits. Better
reconstruction does not automatically mean a better embedding for these tasks.

Earlier plan, completed by RPB-v23: make one separate candidate that explicitly represents
observed candle-to-candle changes while retaining levels and activity. Keep
native32, the current useful branches and the independent fixed heads. Compute
changes only when both adjacent values are observed; propagate that mask and
exclude future values. Use a new design tag and prospective recipe, leaving
RPB-v22 and its trained instances intact. Target improved trend without losing
volatility/activity/relation. Do not spend the next budget increasing dimensions
or tuning heads. Validate under the same MKT-HIST-2 bank first; broader claims
need a new chronological cohort afterward.

The focused [data contract review](DATA_CONTRACT_REVIEW_2026-10-14.md) passes:
timestamps, masks, role containment, zero future and exact H128/H256 recent
suffixes. Legacy graph/runtime orchestration remains outside this scoped audit.
All [25 portable pairs](MARKET_STUDY_BACKPORT_CHECK.json) remain equal across
repositories; these application-only runners/helpers introduce no pending
encoder backport. Market data, loading and evaluation stay in integration.

## Preserved first-pilot context

The [MKT-HIST-1 report](MARKET_HISTORY_PILOT_RESULTS.md) records completed
streaming TRAIN fitting, standard AdamW CUDA training, typed resume states and
encoder-independent fixed readouts. The admitted **RPB-v22.market-01** capsule
is `output/runs/market-history-pilot/run-Ova1QQ`, with unchanged C3/H128/F9/W64/D32
architecture and512 updates. The `.init` label identifies its0-update control.
Keep this model, all controls, original synthetic instances and the unused
January29–31 dates. No forecast/trading task, TEST score or market serving
default was selected.

Volatility/activity linear R² are0.775/0.837, past relation0.405 and past
trend−0.172; raw trend is0.558. These are separate continuous tasks with100%
coverage, not one accuracy percentage. Neural results are mixed and all fixed
repetitions are retained. Fixed-query validation MAE improves0.9431→0.3196.
This establishes pipeline feasibility and some accessible history information,
not a universally good market representation.

Next: define one separate candidate/card for consecutive-candle changes while
retaining level information, masking, native32 and the fixed readouts. Preserve
this control's volatility/activity gains while targeting its weak trend result.
Keep the known development split labelled honestly; broader quality claims need
fresh chronological sources later. Do not spend the next budget tuning heads
or import full Hero/Runtime/Lattice to solve this representation weakness.
The old MDN expects per-channel `[B,N,C,De]`; global `[B,D]` is not that contract,
so design an explicit consumer boundary before migrating it.

Use [the runner](../code/training/README.md) for exact commands, and
[portable training support](STREAMING_TRAINING_SUPPORT.md) for backported APIs.
Streaming/scaler/resume fixtures pass; actual completed-checkpoint recovery
produces identical final inference/features with zero extra optimizer updates.
The [replay](MARKET_PILOT_ENGINEERING_REPLAY.md) preserves one initial capsule
whose executable metadata failed admission; the corrected same-seed replay is
byte-identical numerically. Do not relabel either archive.

The preparation notes below record the prior integration milestone. Its former
pre-market checklist is completed by the bounded pilot; it is not a request to
rerun that experiment. Model geometry and larger-input limitations still apply.

The [13 October encoder research note](ENCODER_RESEARCH_2026-10-13.md) records
the separate v22 content-only candidate and its passed development gates, plus
the preserved v21 failure. V22 is now the primary application API and the
selection in `config/encoder.conf`; see [the selection record](PRIMARY_ENCODER_SELECTION.md).
It reuses the v20 base. Exact v20 settings remain in `config/encoder_v20.conf`
and all 15 reusable file pairs remain equal after relocating that settings path.
Research and synthetic evaluation continue in the sibling embedding repository.

The input/config milestone was implemented as **RPB-v20**. The primary
**RPB-v22** retains that contract and consumes all nine
kline features, configured channel/history sizes and the requested native
embedding width. The core settings module, model and inference checkpoints
are also available in `cuwacunu_embedding`; see the
[backport record](ENCODER_BACKPORT.md). **RPB-v19** and all confirmed evidence
remain untouched. Generalized engineering checks are in
[CONFIGURABLE_ENCODER_VALIDATION.md](CONFIGURABLE_ENCODER_VALIDATION.md).

Read [the stability review](STABILITY_REVIEW_2026-10-12.md) for repaired cache,
normalization, config, checkpoint and environment edge cases. Keep CSV inputs
immutable while preparing caches. New checkpoint filenames are mandatory;
existing artifacts cannot be overwritten through the typed save API.

Preparation checklist from the earlier milestone:

1. Select instruments, stable semantic IDs, one candle interval, history length
   and embedding size in `config/encoder.conf`. The file's interval 1 is a test
   default, not a chosen market cadence. Use the same H for every loader source.
2. Declare chronological TRAIN/VALIDATION/TEST boundaries and the prediction
   tasks in the independent evaluation layer. Fit the scaler only on TRAIN.
   Use raw loader values (`normalization=none`) with the encoder scaler.
3. Add bounded streaming TRAIN statistics. The current fitted scaler accepts
   a complete training tensor; the mmap loader itself reads only requested
   windows. Avoid collecting a large market dataset in RAM just to fit statistics.
   Backport reusable scaler work and its checks to the standalone encoder.
4. Measure mmap throughput/RAM and actual GPU memory on the chosen H/batch size,
   then train a new tagged instance. Check CUDA parameters, loss, gradients and
   updates. Skip batches with no eligible reconstruction targets. Start with a
   short pilot before a long budget; keep native exports without PCA afterward.
5. Keep synthetic quality claims separate. Longer H changes harmonic frequencies
   q/H; nine-feature means also change spectral arithmetic. Evaluate the declared
   market representation rather than assigning v19's synthetic scores to it.

All channels currently need the same complete uniform timestamp grid.
Heterogeneous intervals, unequal source histories and leading timestamp padding
are rejected; resampling remains an explicit future application decision.
H is fixed per model/checkpoint, not dynamic between calls. No maximum tensor
capacity is established: attention grows roughly with `(ceil(H/P))²`, and the
spectral bank with channel pairs × F² × H × eight bins. H257 is a checked shape,
not a maximum. No server listens on the reserved container port yet.
Current continuation checkpoint,15 October2026: the TRAIN scale diagnosis,
v26 and v27 are complete and preserved. V27 passes5/9 original and2/8 strict
conditions, with all three neural trend fits negative; stop its exact map.
The mirrored search ledger keeps eight entries and40 checkpoint pins, v22
primary, native36 balanced and every task/head reference. All73 portable pairs
match. The separately sealed raw-F64 symmetric-relative v28 card is pending
engineering admission, calibration and quality. Follow the current
sections at the start of this document; earlier milestone proposals remain
historical context. No new9/9 winner, primary change or publication is recorded.
