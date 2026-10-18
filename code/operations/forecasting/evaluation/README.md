<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../LICENSE. -->
# Saved-array latent dynamics evaluation

The explicit `versions/v1/` implementation contains deterministic forecasting
controls, fixed-coordinate error/coverage/overlap summaries, diversity and
retention arithmetic, and matched future-risk ridge readouts. It depends on
Torch and the standard library, with no Encoder or market-loader imports.

Point forecasts are compared only within one frozen coordinate system.
Embedding-versus-state comparisons use unchanged observable targets. Numeric
research thresholds and statistical confirmation remain prospective; synthetic
fixtures establish implementation correctness only.
