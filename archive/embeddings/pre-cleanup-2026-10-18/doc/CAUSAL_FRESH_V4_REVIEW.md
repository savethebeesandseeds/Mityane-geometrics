<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# CPB-v4 review and next direction

CPB-v3.market-01 remains the clean baseline and dev selection. CPB-v4.market-01
passes19/22 declared numeric conditions, complete trust admission and100% coverage.
Preserve it as a clean trend/relation and hard-timing reference. It fails overall
advancement; stop this exact recipe without rate, coefficient, seed, budget,
checkpoint or head rescue. No later experiment is running or scheduled.

The [per-task results](CAUSAL_FRESH_V4_RESULTS.md) show both sides of the tradeoff.
Trend improves from0.2631/−0.2540 to0.9072/0.5699 linear/neural R². Relation improves
from−0.0070/−0.0856 to0.3264/0.2095; all three final neural relation and trend fits
are positive. Neural volatility/activity improve too. All eight TEMPO-5 retention
conditions pass, including both input-scaler policies and30% deletion. Validation
query MAE falls0.3764→0.3083 and NLL1.5218→−0.0454. However, linear volatility
falls0.9647→0.8496, linear activity0.8628→0.8351 and TRAIN MAE worsens0.2590→0.2725.
Those three failures remain failures even though other measures improve.

This was one controlled optimization change. A detached change4 feeds only the
conditional variance projection; its108 weights and27 biases still learn.
The mean residual path stays live. Forward capacity, parameter count/native36,
numeric Gaussian NLL, data, paired seeds and2048/B8 budget are unchanged. Encoder
updates intentionally use a mean-path partial NLL gradient, rather than the full
joint Gaussian NLL gradient. The same literal prior and fresh constructor are
used. No pretrained teacher, old learned weights or fitted statistics enter v4.

All13 paired initial embedding/statistic archives are numerically bit exact to
v3, independently verified. The bank ID changes because its hash includes the
versioned TRAIN source identity; numerical counts/moments/scales are unchanged.
This pairing and the explicit gradient proof support the controlled comparison.
They do not establish that variance gradients are universally harmful or that
known-development gains will generalize to another market period. Three head
fits on one checkpoint are not independent encoder training replications.

The next proposed work is a **saved TRAIN-only v3/v4 comparison** of the change4
block, saved fitted-linear contributions and mean/variance gradient paths, focused
on volatility/activity and reconstruction. Use existing predictions, coefficients,
features and query witnesses; the detailed cellwise witnesses cover only the
first8 TRAIN windows per saved stage, while aggregate queries cover112. Do not
use held-out tensors to select a coefficient, add target-shaped features, refit
heads, or replay either encoder. The question is whether the lost linear signal
moved into a different native block, became less linearly accessible, or reflects
the changed optimization pressure. Diagnose before choosing a new mechanism.

Any next encoder needs its own named, prospective declaration, fresh clean
training and unchanged heads. Keep v3's baseline protection and also protect v4's
new scoped gains explicitly; do not forget either result. New-period confirmation
and forecasting remain unmeasured. V1/v2/RPB remain historical references with no
quality floors for clean candidates.

Sources, checkpoints, typed states, embeddings, head states, logs, original
reporting failure and audit repairs are retained locally. The report correction
changes only the reused raw-control metadata predicate and strengthens its hash
binding. Neither model execution nor fitting was repeated. Read the
[training audit](causal_fresh_v4/independent-training-audit.json),
[quality audit](causal_fresh_v4/quality-independent-audit.json) and
[preservation receipt](causal_fresh_v4/final-preservation.json).
New research remains local; this request authorized no commit, push or dev change.
