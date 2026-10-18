# Proposed TEMPO-4 timing and AMP-2 shape challenges

Original status: proposed and unmeasured. The separate
[fixed information card](../code/evaluation/cards/two_component_information_v1.md)
has now implemented this signal family as `two-component-v1` and completed
[data-only information admission](TWO_COMPONENT_INFORMATION_V1.md).
Both tasks recover100% of engineering answers with100% coverage in intact
and extra-deleted views. The [new recipe registry](two_component_dataset_registry.json)
retains the exact source and pre-outcome complexity assignments. The separate
[multiband encoder screen](MULTIBAND_SCREEN_V1.md) is now completed on two
fresh quality cohorts per task. RPB-v19 passes its joint continuation gate;
RPB-v18.alt-01 is the matched new-dataset control. Initial scores credit the
architectural prior; this is development evidence with no promotion.
The proposal below preserves the reasoning; use the frozen
card for exact implementation/draw order. TEMPO-3 and its results remain intact.

**TEMPO-4** names only the slow-band lag-sign task. **AMP-2** names the separate
component-balance task. Prechoose designed complexity **5/5** for each, as an
ordinal label for its proposed two-component structure. These levels are not
measured entropy, accuracy or mathematical comparisons with TEMPO-3 or AMP-1.
The implementation card and separate registry now fix those identities.

## What the completed TEMPO-3 work teaches

The [fresh RPB-v17.alt-01 confirmation](results/fixed_prior_fresh_confirmation_v1.json)
completed all five cohorts: intact Linear/Neural means100%/100%, deleted
82.1875%/99.895833%, with100% coverage. Its all-cohort gate failed because
master84090's deleted Linear score was70.3125%; its initial score was69.53125%.

The [RPB-v18 development screen](results/unit_relation_screen_v1.json) normalized
each fixed four-spacing relation vector to unit length. It passed its targeted
two-cohort gate and completed the other three retained cohorts: intact100%/100%,
deleted100%/99.947917%, with100% coverage. These were the same known datasets,
including the chosen weak cohort, not another fresh-data confirmation. Its
initial control already reached intact100%/100% and deleted100%/99.895833%.
Credit the generic fixed relation calculation and its normalization; this
near-ceiling timing result does not establish useful shape learning from waveform
training or a universal encoder. Neither result implies promotion.

## One coherent signal recipe

Keep C3/H32/F3 and semantic channel IDs 0/1/2. Each independent source contains
two sinusoidal components, with separate phases and well-separated bands:

- slow period P_s uniform on [12,20);
- fast period P_f uniform on [4.5,7);
- phases independently uniform on [0,2*pi);
- slow absolute delay L_s uniform on [.5,1), fast L_f on [.3,.7).

Both components stay below the sampling Nyquist frequency, their bands cannot
exchange identities, and their phase shifts stay away from zero and wrapping.
This is a coherent mixture, not random per-tick labels or chaotic waveform noise.

For channel 0, before nuisance transformations, use

    s0(t) = a_s*sin(2*pi*t/P_s + phi_s)
          + a_f*sin(2*pi*t/P_f + phi_f).

Channel 1 has the same components with their respective signed time shifts.
Channel 2 has an unrelated two-component mixture drawn from the same bands.
All three features are lawful copies multiplied by 1+.1*f. Apply a positive
gain uniform on [.5,1.5) and a constant offset uniform on [-.75,.75) independently
to each channel/feature. The gain multiplies the entire mixture, preserving its
component balance. Keep Gaussian noise SD .005, natural missingness .10, one
three-tick gap per channel across its features, and a separate pair-shared .30
deletion validation view. Do not increase noise or missingness to create an
arbitrary accuracy target.

There are **two separately named datasets/tasks**, with separate balanced
source-paired rows and fixed heads. Never average their accuracies into one
encoder score.

**TEMPO-4 — frequency-specific timing:** draw r uniformly on [.8,1.25), set
a_s=r/sqrt(1+r*r), a_f=1/sqrt(1+r*r), and share these amplitudes across the pair.
Label 0 uses slow shift -L_s; label 1 uses +L_s. The fast shift is xi*L_f, where
xi is an independent balanced sign shared by both variants. Thus a global
signed relation can follow a nuisance component: the task is specifically the
slow band's lead/lag, not the sign of an arbitrary sum. The encoder receives
neither a task-selected pair nor the band, period, delay or label.

**AMP-2 — component balance:** draw r uniformly on [1.8,2.2). Label 1 uses the normalized
amplitudes above; label 0 replaces r with 1/r. This exchanges slow/fast strength
while keeping a_s^2+a_f^2=1, preventing a deliberately different total component
energy from becoming the label. Both lag signs are independent balanced nuisance
variables shared by the pair. Arbitrary positive whole-mixture gains make
absolute magnitude unreliable; relative component shape remains observable.
Finite-window energy can still differ, so do not claim every magnitude shortcut
is eliminated.

For each task, split independent sources into TRAIN128 and VALIDATION64 pairs
before forming variants. Randomize opposite-label row order. Share phases,
frequencies, gains, offsets, noise and masks within each pair; masks and ordering
must not disclose its label. Keep source IDs disjoint across tasks, splits,
engineering and quality seeds. Save only legal observations, masks, scoring-only
labels and source IDs, with hidden storage exactly zero. No hidden waveform or
true generator parameter belongs in encoder fitting, targets, preprocessing or
heads.

## Establish observability before encoder quality

The TEMPO-3 determinant identity does not isolate a component of a mixture.
Cancellation of a summed margin must not redefine a physical label or become
label noise. Use a separately frozen, observed-only information diagnostic.

Proposed diagnostic: estimate the two separated frequency bands jointly from
legal observed values with per-channel constant offsets and sine/cosine
coefficients. Use one bounded frequency-search procedure fixed before data
generation, not the hidden frequencies. Derive the slow inter-channel phase
sign for timing, and within-channel slow/fast amplitude ratio for balance.
Inputs exclude labels and generator truth; labels are used only for scoring.
This fitting is data-only information admission, never an encoder feature,
training target or replacement fixed head.

Freeze support, conditioning, residual and phase/ratio-margin rules first.
Insufficient observations, singular fits or ambiguous margins must abstain and
remain in the total population. No claim of a perfect oracle is made here.
On two independent engineering cohorts, require each task to achieve at least
99% supported accuracy and 99% coverage intact, and 98% accuracy and 95% coverage
under extra deletion. Check noiseless affine cases and masks independently.
If admission fails, retain that recipe and failure. Any revised data recipe
needs a separate prospective identity and admission; supplying hidden frequencies
or dropping failed sources cannot make the original benchmark pass.

## Minimal architecture screen after admission

Freeze separate dataset/task identities and a comparison card before quality.
Start with two fresh cohorts per task, all results retained. Compare unchanged
RPB-v18, its own initial0 export, and one
separately specified frequency-sensitive architectural candidate with its own
initial control. Preserve the generic relation prior: it remains an explicit
block or retained comparator, rather than discarding the useful TEMPO-3 result.
No particular new implementation is selected by this note. Fit these new-dataset
instances and their TRAIN maps/heads on their declared TRAIN data; TEMPO-3 scores
are motivation, not reusable new-dataset controls. Do not rerun an exhaustive
set of older designs.

Use native32, original waveform Huber1, context .15, B8 and512 CUDA updates.
Keep Ridge1 and tanh16/Adam .01/100 with all three fixed repetitions. No
post-encoder PCA, hidden-parameter auxiliary, larger head or selected checkpoint.
Fit heads on each task's TRAIN, reuse them for intact/deleted views, and retain
unsupported fits. Report both tasks, both views, initial/trained scores, mean
and worst cohorts, coverage, query MAE and training cost separately. A practical
continuation gate can retain the existing75% in both heads/both views on every
starting cohort, with100% coverage, applied separately to both tasks; it is not
promotion. Stop weak designs without coefficient or head searches.

High timing accuracy alone would still credit a structural prior. AMP-2 can also
be accessible to the fixed relation bank, so its separate name does not force a
learned-shape solution. Retain each initial control. For RPB-v18, verify the fixed
odd12 coordinates are identical at0/512 on the same inputs; a trained improvement
in component balance can then support additional learned-shape utility under the
fixed heads. If the initial control is already equally strong, report another
prior success, not evidence that waveform training learned the task.
Even success on both tasks establishes only this bounded two-band family, not
universal timing, arbitrary mixtures or robustness outside the declared bands.
