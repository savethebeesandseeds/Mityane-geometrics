<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v5.market-01: separate mean and uncertainty coordinates

Declared16 October2026 before build, calibration, training or quality. The
[machine card](CAUSAL_FRESH_V5_CARD.json) is authoritative. CPB-v3.market-01 remains
the clean baseline and selected dev encoder; v4 remains a scoped clean reference.
All previous sources, results and local ignored checkpoints are protected.

The [saved TRAIN diagnosis](CAUSAL_V3_V4_TRAIN_DIAGNOSIS.md), independently audited,
motivates a new architecture hypothesis. V4's change4 spreads across more signals
and loses linear volatility strength; TRAIN activity improves. This does not
prove a capacity cause or explain its held-out activity regression.

Keep the causal parent32 and mean4 reconstruction route. Add a distinct freshly
learned bank270→uncertainty4 projection. The mean decoder reads mean4; the variance
decoder reads live uncertainty4. Export all40 coordinates directly as
learned20 + literal spectral12 + mean4 + uncertainty4. Gaussian NLL stays the
actual training objective, including its error/variance mean derivative. Variance
gradients reach the new projection, with no path through mean4 or the parent via
the fixed raw causal bank. This does not remove inverse-variance mean weighting.

There are298640 parameter values:294752 learned in76 tensors and3888 literal fixed
analytic prior values. The added1080 learned values and dimension-dependent head
sizes/seeds are explicit capacity differences. No trained parent-prefix preservation
is claimed. Every learned value is freshly initialized, with the new projection
drawn after all inherited constructor draws; source_updates=0. No old checkpoint,
teacher, optimizer or fitted statistics initializes this candidate.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January2024 minute-window reconstruction
and past-signal retention · **complexity4/5**. Keep4096 TRAIN anchors January1–21,
recent128/C3/F9/P8 input, width64/three layers/four heads/feedforward256/decoder128.
Fresh TRAIN-only input/bank calibration uses the unchanged numerical recipe,
zero encoder calls and no VALIDATION statistics. Keep112 TRAIN/32 known chronological
VALIDATION probes and the saved1/16 fixed query plans. January29–31 and new periods
remain reserved and unmeasured.

One trajectory:2048 successful CUDA:0 updates, B8, maximum4096 attempts; AdamW
0.001/decay0.0001/clip1/dropout0, query25%/context deletion15%, seeds943001–943004.
Save inference0/2048 and typed TRAIN512/1024/2048; resume the actual trajectory
at512 without replay. Report synchronized training compute separately from
calibration, proofs, extraction, head fits and I/O.

Before quality, require full masking/causal-cutoff proof for every native40
coordinate before/after engineering learning and on actual saved initial/final
artifacts, including future values/masks/nonfinite poison and positive past
interventions. Require actual CUDA movement in all76 learned tensors, frozen prior
and statistics, typed layout/policy refusal, exact optimizer/counter resume and
hash-bound source/compiler/input/checkpoint closure. A nonzero CUDA fixture must
show live mean-only gradients into mean4, live variance-only gradients into
uncertainty4, zero cross-path gradients, both heads/new projection learning and
manual full-NLL native40 derivative parity.

Both saved stages get new embeddings and new unchanged heads; no PCA after the
encoder, head tuning, intermediate selection or old encoder/head replay. Market
uses joint four-output ridge1 and D40→16tanh→4, Adam0.01/100 updates;164/724 fitted
parameters. Repetitions2701/2802/2903 use unchanged dimension stream seeds
6049515622977958931 /8262941649267994858 /2095371932330426401. These are fixed head
variation on one encoder, not three encoder replications.

Dataset: **TEMPO-5** · nine-feature slow/fast lag-sign timing · **complexity5/5**.
Use saved cohorts941101/941102/941103, each512 TRAIN/256 VALIDATION examples from
256/128 paired source groups, with zero synthetic encoder updates. Score intact
and extra30%-deleted views under fresh synthetic-TRAIN and fresh market-TRAIN
input scalers. Same binary ridge1/tanh16/Adam0.01/100 recipes and D40 seeds;
82/690 fitted parameters. Fresh cohort scalers are shared between saved stages.

Overall advancement requires all22 conditions in the already frozen
[search-goal protocol](clean_causal_encoder_search_goal.json), plus trust and100%
coverage. Each market/TEMPO metric protects the stronger final v3/v4 result;
TRAIN MAE≤0.25904599043427573, VALIDATION MAE≤0.30831692401624544 and VALIDATION
NLL<−0.04538148809081624, with declared1e−12 tolerances. All three trend neural
R² values and both relation R² heads must be positive. No task averaging or
inherited v1/v2/RPB oracle floor. Preserve all gains/failures regardless of outcome.

Use only the existing managed GPU cuwacunu container and exclusive source/build/
run leaves. Do not change the dev selection or concurrent Latent Roadmap work.
No commit/push or promotion is authorized here. Stop this exact recipe after its
fixed final measurement; any further mechanism needs a separate prospective
declaration. The user-authorized research goal remains active across candidates.
