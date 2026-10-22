# V18: contiguous training targets, unchanged evaluation

This is the prospective **1024-update** comparison of contiguous original-grid
target spans against scattered-patch training. The [protocol](PROTOCOL.md) and
[Operation Contract/Requirements-and-Evidence map](CONTRACT.md) were saved before
this study's engineering/model execution and new readouts. No quality result
is established by these prospective documents. The [completed result](RESULTS.md)
and [compact record](../../records/CPB-v18.span-001.json) now report a valid
negative comparison: all six arithmetic audits pass, but every continuation
condition fails. Median paired R² changes are trend −0.221846, activity
−0.090620, volatility −0.057857 and relation +0.015733. V16 remains the reference;
no new scoped linear record or top-three group changed.

V18 changes how the encoder is trained. Architecture/native36, coefficient0.1
Gaussian-plus-hidden-temporal objective, TRAIN fitter/data, row sampler and
optimizer remain shared. At H128/P8 each complete channel hides four consecutive
patches rather than four shuffled positions. Keep masks independent across
channels, original positions intact and missing/partial support explicit.
There is no silent scatter fallback when a legal span is unavailable.

Evaluation stays the same: existing112 TRAIN/32 known January validation probe
archives, original legacy target arithmetic, TRAIN-standardized ridge1,
independent scalar audit, all-arm reporting and the existing trend/activity/
volatility continuation conditions.1024 is the sole primary endpoint. No256
gate, intermediate head scoring or outcome-dependent budget decision is used.

Reuse all three completed V17 coefficient0.1/scattered-control typed256 states
and their exact original executable/settings/data bindings. Continue each
256→640→1024; fit no control preprocessing again. Each candidate starts fresh,
fits the same1024 TRAIN windows, trains0→512 and resumes512→1024. Both arms use
seeds947101/2/3 and the same row/mask/context offsets, with context deletion none.
Require matching same-seed initial/scaler/bank numerical identities. Coefficient
0.03 V17 candidates are preserved and not extended.

Successful completion means six logical1024 trajectories: **6144 logical
updates, including768 already executed control-prefix updates;5376 new physical
scientific updates**. Preserve these distinct costs, twelve planned new
training invocations, actual attempts, fits, failures and separate engineering
work. All six final1024 models seal before new extraction/readout fitting.

The inspected implementation history did not establish a contiguous-target
training comparison; original RPB guidance already mentioned spans in general.
The hypothesis is not a novelty claim or proven explanation of earlier losses.
January/February/March are consumed evidence. This is neither fresh confirmation
nor the historical2048 budget. **V16 seed946103/final2048 remains the research
reference.** A later bounded result cannot automatically change it.

Construction, supported geometry, TRAIN fitting, the complete mask/objective/
optimizer/counter method, typed resume/export and their provenance remain part
of the reusable encoder component. Applications retain their own training and
final docking. Numerical work stays C++ with managed-container Node/Bash
orchestration; no Python or host toolchain is introduced.
