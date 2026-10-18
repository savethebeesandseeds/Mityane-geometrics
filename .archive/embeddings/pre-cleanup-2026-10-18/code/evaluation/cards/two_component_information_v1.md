# Two-component legal-information admission v1

Status before execution: prospective, data-only engineering. Freeze this card,
the complete source closure and verified SDK proof before generating cohorts.
No encoder, classifier, optimizer, PCA, TEST or quality cohort runs here.
The existing TEMPO-3 result and every saved model remain unchanged.

## Dataset identities and physical labels

Shared generator recipe: `two-component-v1`, C3/H32/F3, semantic channels0/1/2.
**TEMPO-4** is `slow_lag_sign`, designed complexity **5/5**. **AMP-2** is
`component_balance`, separately designed complexity **5/5**. Levels are ordinal
design choices made before outcomes; they are not mathematical difficulty or
accuracy measurements. Keep both tasks separate in every result panel.

For each source, slow period Ps is U[12,20), fast period Pf is U[4.5,7), with
independent phases U[0,2*pi). Slow positive delay Ls is U[.5,1), fast Lf is
U[.3,.7). Channel0 is the sum of slow and fast sinusoids. Channel1 contains
the same components with their independently signed shifts. A shifted component
is sin(2*pi*(t+signed_shift)/P+phase). Positive slow shift is timing label1;
negative is label0. This explicit convention avoids changing labels based on a
summed analytic margin.

TEMPO-4 draws r U[.8,1.25), amplitudes (r,1)/sqrt(1+r*r), shared by both label
variants. The slow shift changes from -Ls to +Ls between label0/1. A fair fast
shift sign is shared by the variants and is a nuisance, independent of label.

AMP-2 draws r U[1.8,2.2). Label1 uses (r,1)/sqrt(1+r*r); label0 uses the same
formula with 1/r, exchanging relative slow/fast strength while preserving the
sum of squared component amplitudes. Fair independent slow/fast shift signs
are nuisance variables shared by both variants. Finite-window energies can
still differ; no claim that all magnitude shortcuts are eliminated.

Channel2 has independent Ps2/Pf2/phases from the same bands and r2 U[.8,1.25),
with the same normalized-amplitude formula. It is unrelated, pair-shared and
label-independent for both tasks. Every channel/feature has an independent
positive gain U[.5,1.5) multiplied by 1+.1*f and offset U[-.75,.75), constant
over time and shared within the pair. Add Gaussian noise SD .005 after affine
transformation. Natural coordinate missingness is .10. Each channel has a
three-tick gap across its three features, start floor(30*U), ticks0..29.
Gains, offsets, noise, masks and gaps are source-paired. Hidden storage is zero.

## Reproducible source and nuisance streams

All engines are local std::mt19937_64; real uniforms are high53 bits divided
by2^53. Use the unchanged shared `stream_seed` function. Task streams are
`0x7463342d6c616731` for timing and `0x616d70322d62616c` for balance.
Namespaced = stream_seed(master,task_stream).

Role streams: split `0x74632d73706c6974`, signal `0x74632d7369676e6c`,
distractor `0x74632d6469737472`, affine `0x74632d616666696e`, noise
`0x74632d6e6f697365`, mask `0x74632d6d61736b31`, gaps `0x74632d6761707331`,
order `0x74632d6f72646572`. Per-source role seed is
stream_seed(stream_seed(namespaced,role),original_source_index+1).

Assign sources before variants using descending Fisher-Yates, with unbiased
full64 bounded rejection threshold=(-bound)%bound on the split engine. The
first128 shuffled sources belong to TRAIN, the remaining64 to VALIDATION.
Emit each split in ascending original-source order. Pair row order uses its
own one-uniform fair draw, row0label=U>=.5 ?1:0, row1 the opposite.

Exact per-source draw order: signal Ps,Pf,phase_s,phase_f,Ls,Lf,r,slow_sign,
fast_sign (the timing task replaces slow_sign with its label); distractor
Ps2,Pf2,phase_s2,phase_f2,r2; affine channel/feature gain then offset; gap
starts channel0..2; noise channel/time/feature with two draws per cell,
sqrt(-2*log(1-U1))*cos(2*pi*U2); natural masks in the same lexical cell order;
then the independent row-order draw. No generator truth is exported.

Master must be <2^32; total sources must be <2^30. Reject bounds before draws
or allocation. Packed signed Long source ID is (master<<31)|(source_index<<1)|
task_bit, timing0/balance1. These admitted IDs are injective across sources,
tasks and masters. Every source appears exactly twice with opposite labels.

Deleted VALIDATION is a pair-shared additional .30 coordinate-erasure view.
For each packed ID, its engine seed is
stream_seed(stream_seed(master,0x74632d64656c7631),ID); erase when high53 U<.30
in channel/time/feature order. Retain rows, IDs, labels and observed subset
values exactly. This additional view does not change dataset identity.

## Observed-only information rule

Rule `observed-two-band-affine-fit-v1` receives only observations and masks.
Scoring labels and source IDs are absent from its interface. It estimates shared
slow/fast frequencies from legal channel0/1 values; channel2 is irrelevant to
this separate task-information check. It is not an encoder feature or target.

For each eligible feature, fit independent coefficients for constant, slow
sine/cosine and fast sine/cosine at candidate frequencies. Choose frequencies
by the aggregate observed residual. Search the full fixed slow[1/20,1/12] and
fast[1/7,1/4.5] bands on a17x25 grid, then10 deterministic3x3 local refinements:
at most515 frequency pairs per row. The implementation fixes tie order, clipping
and refinement step schedules. No true frequency or label enters the search.

Abstention thresholds are frozen before engineering outcomes:

- Each eligible feature has at least10 observed ticks; each channel has at
  least2 eligible features, at least16 distinct observed ticks and span>=24.
- Scaled5x5 Cholesky pivot floor1e-8; energy floor1e-12.
- Normalized residual<=.0025; each component's energy fraction>=.05.
- Within-channel phase coherence>=.95 for both components.
- Timing: wrapped slow phase difference, positive predicts1; absolute margin
  in[.08,.8] radians.
- Balance: mean log(slow/fast fitted amplitude), positive predicts1; absolute
  margin in[.35,1.0], per-feature log-ratio spread<=.1.

Every unsupported row remains in the total population with its reason. No
perfect-oracle claim or dropped-source correction is allowed. The fixed
diagnostic's bands reflect the known data family; an encoder receives no
source-specific band, true period, delay, label or analytic fitted feature.

## Execution, acceptance and failure retention

Engineering masters are exactly910901 and910902, independent of previous
quality runs. For each master and task generate TRAIN256/128 paired sources
and VALIDATION128/64 sources once. Score TRAIN, intact VALIDATION and deleted
VALIDATION separately. There are four generator calls,12 data archives,
12 decision archives and2,048 data-information rows; no quality seeds allocated.

On **each task, master and view**, require >=99% supported accuracy and >=99%
coverage on TRAIN/intact VALIDATION; >=98% accuracy and >=95% coverage on
deleted VALIDATION. All conditions must pass before encoder quality runs.
These are information gates, not classifier-head accuracy or promotion.

First run explicit generator invariants and noiseless affine/masked information
fixtures, including abstentions and invalid inputs. The noiseless generator API
is engineering-only and never substitutes for the noisy data recipe. Build/run
inside the existing managed container and isolated named session. CPU scalar
fitting is this data-only diagnostic; it executes no encoder or optimizer.

Save exact source/card/SDK pins, legal observations, masks, scoring-only labels,
packed source IDs, every supported/abstained decision, frequency/residual/margin
diagnostics, selected-fit coefficients in original observation units, eligible
feature flags, bounded search counts, costs, logs and completion receipt.
Independently check selected-frequency normal-equation stationarity, residual,
phase/ratio margins and support using those witnesses, plus
CPU archives, populations, masks, source disjointness, deletion streams and
score/gate arithmetic without rerunning frequency fitting or any model.

If admission fails, preserve the failure and stop encoder quality for this
admission. Any changed information rule needs a new prospective rule/card and
separate evidence; any changed signal recipe needs a new recipe identity before
generation. Never provide hidden frequencies, redefine labels, discard failed
sources or silently tune support thresholds after seeing these cohorts.
