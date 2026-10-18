# TRAIN support-component audit

This is an **exploratory diagnosis selected after the known RPB-v21.alt-01 quality outcome**, completed on 13 October 2026. It is not a prospective quality experiment or a causal explanation of the failed deletion gate.

[analysis.py](analysis.py) preserves the exact Python script first executed through container stdin. Its saved-file replay reproduced the same result. [report.json](report.json) contains that output under `scalar_analysis`, the script receipt, exact input hashes, sealed historical/candidate manifest bindings, and the frozen generator source binding.

## Scope and arithmetic

The script uses the existing lazy stdlib archive reader and selects only two original TRAIN masks, the two trained relation-projection matrices, and saved TRAIN native32 exports. It summarizes the first 20 native coordinates. It selects or decodes no numerical VALIDATION field. Whole-file hashes cover archives that also contain unselected validation storage.

There are zero encoder, decoder or classifier calls, zero classifier fits and zero optimizer updates. No torch runtime is imported. The audit runs inside the existing managed embedding container. The frozen model, protocol, card and sealed capsules are untouched.

For C3/H128/F9, the last 972 columns of each 20x2916 projection multiply the support fractions `q=n/(H-lag)`, using lags 1/2/4/8 and the original canonical pair/feature ordering. Integer bit intersections reconstruct four-endpoint common counts from each TRAIN mask. The script computes `W_support*q` using saved float32 weights, Python float count fractions and `math.fsum`.

These contribution values are **scalar numerical estimates, not a bit-exact CUDA bank or float32 matrix-multiplication replay**. The exact within-pair equality follows from identical masks and weights. Mean/RMS comparisons do not establish causal decoder use, information content or classifier failure.

## Findings

Dataset: **TEMPO-5**, slow cross-channel timing beneath fast rhythms and structured distractors, nine grouped features; designed complexity **5/5**. These are full-context TRAIN diagnostic quantities, not accuracy.

| TRAIN cohort | Support fraction mean | Support contribution RMS | Centered contribution RMS |
| --- | ---: | ---: | ---: |
|941101|0.57720|1.78405|0.07366|
|941102|0.57630|2.77178|0.08538|

RMS averages squared contributions over 512 TRAIN rows and 20 learned coordinates. Centered RMS subtracts each coordinate's TRAIN mean first. All 19,440 support-block weights are nonzero in each checkpoint. The support contribution is exactly equal within all 256 timing pairs in each cohort: it cannot directly separate those paired variants.

The contribution is chiefly a mask-dependent offset. This makes it a useful mechanism hypothesis for the separately observed deletion failure; the audit does not attribute that failure to it.

## Deletion-policy binding and limits

The original sealed generator's `main.cpp:57` calls `delete_observations(validation,.30,seed)`. Its frozen `configurable_capacity.cpp:147-166` draws `uniform53(rng)<probability` separately for every time/channel/feature coordinate and reuses one erasure mask within each source pair. It does not select an exact deletion count without replacement. `uniform53` uses the upper 53 bits of `mt19937_64`.

Under the ideal independent-Bernoulli interpretation of that policy, a four-endpoint common observation survives with probability `0.7^4=0.2401`, up to negligible 53-bit threshold discretization. Consequently `E[W_support*q_deleted | full mask]=0.2401*W_support*q_full`. This is an expectation over possible erasures, **not the realized factor in the saved deleted view**.

The report's expected contribution-change RMS values 1.35570/2.10628 derive solely from TRAIN masks and weights. No deleted view was simulated or decoded. Odd/even moment estimates can also change with deletion, and the complete classifier can respond to interactions among learned and fixed coordinates. This audit cannot isolate those effects.

All eight selected historical artifacts, including the two frozen generator source files, match their sealed manifests. The scalar script independently checks its selected source/input bytes before and after calculation. This study did not repeat either entire historical inventory; their original complete receipts remain preserved.

## Reproduction and continuation

From the existing managed container:

```bash
source /etc/profile.d/embedding.sh
cd /embedding
python3 -B doc/results/configurable_learned_relation_v1/train-support-audit/analysis.py
```

The command prints the `scalar_analysis` object; the report additionally contains the external sealed-manifest/source-policy receipt. The exact preserved script has SHA256 `a5634ddaf4b13ebf91db78c6e3bca6778a23a9b88635fc4a500681187f6550b9`.

The next proposed experiment is a separately tagged **content-only ordered-relation architecture**, provisionally RPB-v22: keep all odd/even moments, lags, feature pairs and validity/energy guards, but use support counts only for bookkeeping and normalization rather than additive learned-projection inputs. It would keep native32/fixed12, the decoder, loss, masks, optimizer, budget and heads. This is prospective; this audit implements or authorizes no experiment and changes no serving default.
