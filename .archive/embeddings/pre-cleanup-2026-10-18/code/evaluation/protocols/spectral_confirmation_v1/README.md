# Fresh spectral confirmation v1

The [prospective card](../../cards/spectral_confirmation_v1.md) fixes a separate
**RPB-v19.alt-01** instance bundle: unchanged RPB-v19 on three new cohorts per
task. **TEMPO-4** tests slow-component lead/lag; **AMP-2** tests relative rhythm
strength. Both have designed complexity **5/5** and remain separate.

Each trajectory retains initial0/trained512 native32, its TRAIN-only scaler,
fixed heads, waveform queries, full losses and CUDA checkpoint. The18 legal
input archives are generated once, then bound before model construction.
The absolute gate requires both heads>=75% and coverage100% for every
task/master/view, with no adaptive seed extension or rescue. Preserve all
results even if it fails. Initial performance credits the fixed spectral prior.

Run only inside the documented managed development container:

```sh
bash code/evaluation/protocols/spectral_confirmation_v1/admit.sh
# Substitute the exact new admission path printed above, once:
bash code/evaluation/protocols/spectral_confirmation_v1/run_confirmation.sh /embedding/output/runs/rpb-spectral-confirmation/admission-XXXXXX
python3 -B code/evaluation/protocols/spectral_confirmation_v1/publish.py --admission /embedding/output/runs/rpb-spectral-confirmation/admission-XXXXXX
```

The encoder implementation, shared evaluation helpers, generic compiled objects,
SDK and generator are reused. Historical protocol sources have hardcoded
identities, so this entry point retains their loop and arithmetic in a separate
closed protocol rather than changing the old SOURCE. CPU saved-only checking
replays retained fitted heads and query arithmetic; all encoder operations use
CUDA. No v18/raw refit, information search, PCA, TEST or reference promotion.
