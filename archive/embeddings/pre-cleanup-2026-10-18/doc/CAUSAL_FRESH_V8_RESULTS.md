<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v8.market-01 results:4/22, two new scoped references

**CPB-v3.market-01 remains the clean baseline and sole dev encoder.** V8 passes
4/22 prospectively strengthened V3/V4/V5/V6/V7 retention conditions,100% coverage.
It sets neural-volatility R²0.7359692462937604 and market-scaler intact neural
timing accuracy0.9926215277777778 as new scoped references. It fails18 conditions.
Preserve these gains and every earlier scoped leader; no overall winner or dev
replacement follows.

**Measured candidate: independent audits and final evidence join passed.**
Training audit passes310003 checks (21443 core +61153 statistics +1151 saved
state +117864 Gaussian +69301 actual components +39091 temporal learning).
Quality audit passes12643, both with zero mismatches. All13 declared initial
DATA/statistic pairings versus V6 match exactly: eight feature archives compare
only native coordinates20:40 plus their other declared fields; five statistic
archives compare their whole declared statistics. Learned content0:20 is
expected to differ. Whole-native40 equality is not claimed. Frozen producer
trust-pending fields remain exact. The separate
[final evidence join](causal_fresh_v8/final-establishment.json) now confirms full
trust, measured-candidate admission, all13 partial pairings and coverage.
Advancement remains false at4/22; no winner or application selection is made.

V8 changes one structural mechanism relative to V6: four original-position
temporal bins replace one learned temporal pool. Each bin pools its own valid
original patch positions and supplies eight disjoint rows of the same32-row
export projection. Both ordinary and early-contextual paths use the same owned
kernel. Empty-bin output is finite and exactly zero after projection bias.
Concatenated slices feed the unchanged channel/global99→64→32→20 content route.
This changes information allocation without changing registered tensor shapes.

Native40 remains20 learned content +12 literal spectral +4 learned mean
+4 learned uncertainty. All294752 learned values across76 tensors and fresh
TRAIN-only input/bank statistics start from scratch, without old model/stat
import. The3888-value literal prior stays fixed,77 tensors/298640 values total.
Seventy-five identifiable tensors show nonzero CUDA data-gradient and movement
events; their294751 values describe capacity, not every scalar having a nonzero
gradient. The remaining pool-score bias is a softmax gauge, separately observed
and still registered in AdamW; decay/movement does not establish data learning.
All four export row groups and four within-bin centered position contrasts
satisfy the finer prospective learning requirements.

The V6 recipe remains: unit Huber(delta1) mean supervision plus detached-error
Gaussian variance calibration, shared global norm2 clip1, one AdamW, same
constructor draws/initialization943001, streams943002/943003/943004,2048/B8 and
fixed heads. The actual CUDA:0 trajectory completes2048 successful updates
without skips, with exact-state typed resume at512 and real continued training.
The main run makes no unobserved counterfactual next-step bit-parity claim; a
separate engineering fixture tests synthetic next-step resume parity.

MKT-HIST-2 uses4096 TRAIN anchors and112 TRAIN/32 known chronological VALIDATION
readout windows: Jan1–21 TRAIN, Jan22–28 VALIDATION, Jan29–31 unused. BTC/ETH/SOL
minute windows share one historical period, not independent market cohorts.
Targets retain past-window signals, not forecasts. Fixed heads are ridge
penalty1 and tanh16/Adam0.01/100 updates; repetitions2701/2802/2903 use D40 seeds
6049515622977958931,8262941649267994858,2095371932330426401. Market heads have
164/724 fitted parameters, TEMPO82/690. V3/V4 native36 uses148/660 and74/626 and
its declared width-derived seeds; these width/capacity/seed confounds remain.

TEMPO-5 cohorts941101/941102/941103 each contain512 TRAIN/256 VALIDATION examples
from256/128 source groups. Both TRAIN-fitted synthetic and saved market scaler
policies, intact and extra30% coordinate-deletion views, remain unchanged.
Paired examples share a source. V8 initial0/final2048 embeddings and their own
fixed heads are newly measured. V3–V7 final scores/costs below are reused
unchanged. No new V3–V7 experiment, old encoder/head replay or refit. Three
head fits are variation on one encoder/split, not three encoder runs or
fresh-period uncertainty. No post-encoder PCA, best-head/intermediate-checkpoint
selection or head tuning is used.
Complexity4/5 and5/5 are ordinal recipe labels, not accuracy or observed task difficulty.

## Volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_volatility; V3/V4/V5/V6/V7 final rows reused, V8 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.964673434 | 0.463051923 | 100 |
| CPB-v4.market-01 | 36 | 0.84956221 | 0.555176229 | 100 |
| CPB-v5.market-01 | 40 | 0.95221109 | 0.648190685 | 100 |
| CPB-v6.market-01 | 40 | 0.877120189 | 0.58463377 | 100 |
| CPB-v7.market-01 | 40 | 0.880509577 | 0.673304976 | 100 |
| CPB-v8.market-01.init | 40 | 0.0415367359 | -0.354813392 | 100 |
| CPB-v8.market-01 | 40 | 0.907581398 | 0.735969246 | 100 |


## Trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_trend; V3/V4/V5/V6/V7 final rows reused, V8 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.263132963 | -0.253979001 | 100 |
| CPB-v4.market-01 | 36 | 0.90721343 | 0.569875231 | 100 |
| CPB-v5.market-01 | 40 | 0.868645445 | 0.523928784 | 100 |
| CPB-v6.market-01 | 40 | 0.675860749 | 0.270702751 | 100 |
| CPB-v7.market-01 | 40 | 0.685042973 | 0.10961324 | 100 |
| CPB-v8.market-01.init | 40 | -0.0708018923 | -0.140200382 | 100 |
| CPB-v8.market-01 | 40 | 0.69630659 | -0.0374078453 | 100 |


## Activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_activity; V3/V4/V5/V6/V7 final rows reused, V8 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 0.862840311 | 0.541426056 | 100 |
| CPB-v4.market-01 | 36 | 0.835143075 | 0.687370545 | 100 |
| CPB-v5.market-01 | 40 | 0.844256819 | 0.617983776 | 100 |
| CPB-v6.market-01 | 40 | 0.672769992 | 0.323179998 | 100 |
| CPB-v7.market-01 | 40 | 0.712390048 | 0.352638034 | 100 |
| CPB-v8.market-01.init | 40 | 0.59013045 | 0.574945536 | 100 |
| CPB-v8.market-01 | 40 | 0.819083477 | 0.654256366 | 100 |


## Relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. past_relation; V3/V4/V5/V6/V7 final rows reused, V8 initial/final newly measured.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | -0.00700444683 | -0.0855656715 | 100 |
| CPB-v4.market-01 | 36 | 0.326405067 | 0.209510464 | 100 |
| CPB-v5.market-01 | 40 | 0.252328556 | 0.370505348 | 100 |
| CPB-v6.market-01 | 40 | 0.622100687 | 0.322683609 | 100 |
| CPB-v7.market-01 | 40 | 0.600800572 | 0.324861744 | 100 |
| CPB-v8.market-01.init | 40 | 0.347176956 | -0.114271586 | 100 |
| CPB-v8.market-01 | 40 | 0.504487017 | 0.0389512195 | 100 |


Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_intact; V3/V4/V5/V6/V7 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 96.484375 | 95.703125 | 100 |
| CPB-v4.market-01 | 36 | 98.3072917 | 96.9618056 | 100 |
| CPB-v5.market-01 | 40 | 97.1354167 | 96.5711806 | 100 |
| CPB-v6.market-01 | 40 | 98.6979167 | 98.4809028 | 100 |
| CPB-v7.market-01 | 40 | 98.5677083 | 97.8732639 | 100 |
| CPB-v8.market-01.init | 40 | 95.0520833 | 93.8802083 | 100 |
| CPB-v8.market-01 | 40 | 97.65625 | 97.2222222 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. synthetic_train, validation_deleted; V3/V4/V5/V6/V7 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 91.1458333 | 91.7534722 | 100 |
| CPB-v4.market-01 | 36 | 94.6614583 | 93.359375 | 100 |
| CPB-v5.market-01 | 40 | 92.96875 | 92.7951389 | 100 |
| CPB-v6.market-01 | 40 | 95.3125 | 93.4027778 | 100 |
| CPB-v7.market-01 | 40 | 93.359375 | 92.2309028 | 100 |
| CPB-v8.market-01.init | 40 | 89.7135417 | 86.5017361 | 100 |
| CPB-v8.market-01 | 40 | 93.2291667 | 93.0121528 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_intact; V3/V4/V5/V6/V7 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 99.3489583 | 98.1770833 | 100 |
| CPB-v4.market-01 | 36 | 99.3489583 | 98.828125 | 100 |
| CPB-v5.market-01 | 40 | 99.21875 | 98.8715278 | 100 |
| CPB-v6.market-01 | 40 | 99.7395833 | 99.21875 | 100 |
| CPB-v7.market-01 | 40 | 99.21875 | 98.4809028 | 100 |
| CPB-v8.market-01.init | 40 | 94.7916667 | 92.96875 | 100 |
| CPB-v8.market-01 | 40 | 99.609375 | 99.2621528 | 100 |

Dataset: **TEMPO-5** · slow lead/lag beneath fast rhythms, nine grouped features and structured distractors · **complexity 5/5** · three cohorts, each TRAIN512 / VALIDATION256. market_scaler, validation_deleted; V3/V4/V5/V6/V7 reused.

| Method | Size | Linear head % | Neural head % | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 36 | 94.921875 | 94.2708333 | 100 |
| CPB-v4.market-01 | 36 | 95.8333333 | 95.6163194 | 100 |
| CPB-v5.market-01 | 40 | 94.140625 | 94.0972222 | 100 |
| CPB-v6.market-01 | 40 | 97.265625 | 95.3125 | 100 |
| CPB-v7.market-01 | 40 | 94.53125 | 91.8836806 | 100 |
| CPB-v8.market-01.init | 40 | 88.4114583 | 84.9392361 | 100 |
| CPB-v8.market-01 | 40 | 96.2239583 | 95.0520833 | 100 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL minute history, past-window retention · **complexity 4/5** · TRAIN112 / known chronological VALIDATION32. Fixed saved query masks; standardized mean MAE; Gaussian NLL separate.

| Method | Updates | TRAIN MAE | VALIDATION MAE | GPU training seconds |
| --- | --- | --- | --- | --- |
| CPB-v3.market-01 | 2048 | 0.25904599 | 0.376414231 | 144.341116 (reused) |
| CPB-v4.market-01 | 2048 | 0.272508032 | 0.308316924 | 172.860055 (reused) |
| CPB-v5.market-01 | 2048 | 0.254463936 | 0.375196294 | 134.983334 (reused) |
| CPB-v6.market-01 | 2048 | 0.250172579 | 0.354519735 | 175.575555 (reused) |
| CPB-v7.market-01 | 2048 | 0.241805338 | 0.370420511 | 219.730088 (reused) |
| CPB-v8.market-01.init | 0 | 0.664360632 | 0.958394466 | 0 |
| CPB-v8.market-01 | 2048 | 0.247146261 | 0.324973012 | 178.284957 |

## What changed, limits and next direction

The four passing conditions are neural volatility, market-scaler intact neural
timing and the two relation-positivity conditions. Seven market and seven timing
retention floors fail, along with both fixed-query MAE ceilings, strict known-
VALIDATION NLL and all-three neural trend positivity. V8 trend head repetitions
are0.348734190,-0.875923336,0.414965610; relation repetitions are
0.356560851,-0.689937124,0.450229932. Keep all three; choosing the best fit would
change the frozen recipe. A positive mean relation R² permits one negative fit.

V8 improves activity and known-VALIDATION reconstruction relative to V6, while
missing protected V3/V4 activity floors and V4 validation MAE/NLL. V7's TRAIN
MAE/NLL records remain intact; V8 sets no reconstruction record. Reconstruction
measures the encoder/decoder route, not served-embedding quality. Native40,
counts and readouts match V6; cross-width comparisons retain other confounds.
One fresh trajectory on known development populations does not establish
generalization or prove a particular old pooling failure.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL past-window retention · **complexity4/5** · one fresh2048/B8 CUDA trajectory, separate timing scopes.

| Timing scope | Seconds | Meaning |
| --- | ---: | --- |
| Cumulative synchronized encoder work | 178.284956662 | Forward/objective/backward/global clip/AdamW, including in-step checks |
| Training trajectory wall | 237.149576521 | Includes work/I/O outside timed steps |
| Training wrapper wall | 247.822273838 | Separate orchestration timing |
| Fresh TRAIN-only calibration | 3.685512546 | No imported old statistics |

Loader/mask preparation, checkpoints and evidence logging are outside the step
clock. Quality takes93.45364s separately across MKT-HIST-2 and TEMPO-5. Costs are
descriptive; UTC clock gaps occurred, so do not sum wall timestamps or claim a
matched runtime win. Reused historical costs keep their original scopes.

Engineering264423 checks/7372 CUDA calls/110 typed refusals pass. The dedicated
temporal fixture passes61 checks, and the unchanged composite-objective fixture
passes48 CUDA checks. Each saved initial/final checkpoint passes128 cutoffs,
2350 future value/mask cases and30 internal interventions, state unchanged and
zero extra updates. [DETAIL](CAUSAL_FRESH_V8_DETAIL.md) retains physical errors,
all neural repetitions,22 decisions, checkpoint hashes and complete evidence.

Stop this exact four-bin recipe after2048 updates, preserving its two new scoped
records and every V3–V7 strength. No bin/coefficient/rate/seed/budget/checkpoint/
head rescue. Propose a separately declared saved TRAIN-only V6/V8 comparison of
content-block/readout contributions, matched query patterns and temporal-group
evidence before selecting any further mechanism. That proposal does not
authorize a next encoder. Any future protocol protects every scoped leader
before new candidate quality; TRAIN NLL remains descriptive, without a new gate.

See [REVIEW](CAUSAL_FRESH_V8_REVIEW.md), [the card](CAUSAL_FRESH_V8_CARD.md),
[training audit](causal_fresh_v8/independent-training-audit.json),
[quality audit](causal_fresh_v8/quality-independent-audit.json) and
[the preserved ledger](clean_causal_encoder_research_ledger.json).
The completed final evidence join establishes full trust;4/22 still fails overall advancement. No winner, application selection or new training authorization is made.
