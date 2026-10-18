# Frozen market encoder on hard synthetic signals

Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.

**Yes: the exact saved market-trained RPB-v24.market-01.alt-01 performs strongly
on TEMPO-5, designed complexity 5/5.** This is new frozen transfer evidence,
measured 14 October 2026. It uses one market checkpoint across three existing
synthetic cohorts; no synthetic encoder training or checkpoint selection occurs.

TEMPO-5 contains slow lead/lag beneath fast rhythms, nine grouped features and
structured distractors. Cohorts 941101/941102/941103 each have 512 TRAIN examples
and 256 known VALIDATION examples, with intact and saved additional 30% erasure
views. Geometry C3/H128/F9 matches the checkpoint. B8 CUDA:0 eval/NoGrad
inference exports native36 directly, with no post-encoder PCA or decoder calls.

The checkpoint is integration's
`output/runs/market-separate-changes-native36/run-4SauBf/checkpoints/inference-2048.pt`,
SHA256 `a3939b13df8b1704fc3621876cf3adc6efaddd98951147ff43ec6408978fa98c`.
It retains its market-trained v22 prefix32 and learned change4 coordinates.
All parameters/buffers, configuration, scaler files and observations stay exact.
There are zero encoder updates or scaler refits; labels never enter encoder calls.
Only the declared readout normalizers and heads fit synthetic TRAIN embeddings.

Both input-scaler policies were declared before quality testing. The primary
policy reuses each cohort's original synthetic TRAIN scaler externally to the
immutable checkpoint. Mean linear/neural accuracy is 98.96%/97.74% intact and
93.36%/92.53% deleted. These exceed reused RPB-v22.alt-01 synthetic-trained
native32 means of 97.14%/95.96% intact and 91.28%/91.06% deleted, respectively.
All four primary retention comparisons pass, with 100% coverage throughout.

The secondary policy retains the checkpoint's unchanged market scaler. It
reaches 100%/99.52% intact and 96.09%/95.05% deleted. Both policies are reported;
the secondary result does not replace the predeclared primary comparison.
In the secondary comparison tables the reused v22 row retains its original
synthetic scaler; only the native36 row uses the checkpoint's market scaler.

Readouts remain ridge 1 and neural 16 tanh/Adam .01/100 updates, with the unchanged
three-repetition seed policy. D36 changes parameter counts and width-derived
seeds as declared. Each fit scores both validation views without refitting;
the old synthetic-trained v22 scores are reused without encoder or head replays.
This is measured transfer ability, not an equal-training-budget causal comparison.

The [declared card](../../cuwacunu_dev/doc/MARKET_SYNTHETIC_TRANSFER_CARD.md),
[full results](../../cuwacunu_dev/doc/MARKET_SYNTHETIC_TRANSFER_RESULTS.md) and
[execution verification](../../cuwacunu_dev/output/runs/market-synthetic-transfer/run-oujlisd9/verification.json)
record all cohorts/repetitions, exact identities and separate execution times.
Verification preserves all model state, 91 protected files and 15 saved feature files.

The result covers known TEMPO-5 validation only. The older TEMPO-4 and AMP-2
arrays use C3/H32/F3, so this exact checkpoint was not silently reshaped to
score them. TEST, other signal families and unseen confirmation remain unmeasured.
Keep RPB-v22 primary and preserve the separate market linear-volatility failure.
Native36 is the leading research candidate under the
[saved milestone](MILESTONE_2026-10-14_NATIVE36_TRANSFER.md). The next priority is
the saved-TRAIN volatility diagnosis; it is not measured or implemented here.
