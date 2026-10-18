<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../LICENSE. -->
# Saved TRAIN V4/V5 diagnosis and next mechanism

**The independent diagnosis audit passes 282959 checks and 256 finite differences,
with zero mismatches.** The saved-data
producer passed once, with zero model calls, neural forwards, training updates
or fits. Both saved final trajectories had 2048 updates at batch 8 on CUDA:0;
the diagnosis itself took 1.494 seconds of scalar arithmetic, with no new
training or quality run. No new experiment or held-out analysis was performed.
CPB-v3.market-01 remains selected; preserve every V3/V4/V5 scoped strength and artifact.

The most justified next controlled test is **unweighted mean supervision with
separate variance calibration, retaining V5's native40 geometry**. This remains
an interpretation for review, not a next encoder, source, card, loss coefficient
or training declaration. The evidence identifies remaining inverse-variance
mean weighting and representation redistribution; it does not prove why held-out
scores changed or that the proposed objective will improve them.

The [prospective card](CAUSAL_V4_V5_TRAIN_DIAGNOSIS_CARD.md) admits 41 exact
TRAIN inputs plus 23 preservation pins and the final V5 preservation receipt.
The [saved diagnosis](causal_v4_v5_train_diagnosis/diagnosis.json) is 2079052 bytes,
SHA256 `e4d315a062f7021897476f154ce0bdc572d0e8dfb994fc4840b431dda6194b24`.
The [independent audit](causal_v4_v5_train_diagnosis/independent-audit.json) is
54959 bytes, SHA256
`57845bec8f2a93adb8ad94db94c5e0361e1e1bcde11104b7aa081182af0717e4`.
Its 256 local finite differences have maximum absolute error
`9.490406793766226e-10`; they validate bounded scalar adjoints, not a full
encoder/optimizer replay.
It examines four saved stages: V4/V5 initial0 and final2048. Each has 112
natural-mask TRAIN embeddings and saved predictions for all four tasks and
three neural repetitions 2701/2802/2903. These are in-sample predictions from
already fitted heads, not new quality estimates or independent encoder runs.

The old heads remain ridge penalty 1 and 16 tanh units / Adam 0.01 / 100 updates.
V4 native36 and V5 native40 use their original width-derived seeds and parameter
counts; initial common36 values pair exactly, while initial fixed-head scores
can differ because V5 adds four coordinates and uses wider heads/different seeds.
Fresh ancestry remains source_updates=0; no checkpoint or teacher initializes
anything here. Aggregate query files contain both splits, but only TRAIN
subtrees are accessed. No held-out tensor or numeric field enters this report.

## Saved TRAIN task readouts

### past_volatility

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history retention · **complexity 4/5** · saved TRAIN only. 112 natural-mask TRAIN rows; neural column averages the three saved fits. Coverage is complete.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01.init | 36 | 0.8483028632 | 0.9061798393 | 100 |
| CPB-v4.market-01 | 36 | 0.9565033161 | 0.9376374113 | 100 |
| CPB-v5.market-01.init | 40 | 0.8566119371 | 0.9216185622 | 100 |
| CPB-v5.market-01 | 40 | 0.9871272299 | 0.9596348584 | 100 |

### past_trend

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history retention · **complexity 4/5** · saved TRAIN only. 112 natural-mask TRAIN rows; neural column averages the three saved fits. Coverage is complete.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01.init | 36 | 0.5973270884 | 0.8963026583 | 100 |
| CPB-v4.market-01 | 36 | 0.9513668859 | 0.9590665426 | 100 |
| CPB-v5.market-01.init | 40 | 0.6754968042 | 0.9542596202 | 100 |
| CPB-v5.market-01 | 40 | 0.9410406055 | 0.9595791016 | 100 |

### past_activity

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history retention · **complexity 4/5** · saved TRAIN only. 112 natural-mask TRAIN rows; neural column averages the three saved fits. Coverage is complete.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01.init | 36 | 0.9462366278 | 0.9496608597 | 100 |
| CPB-v4.market-01 | 36 | 0.9746126215 | 0.9687507362 | 100 |
| CPB-v5.market-01.init | 40 | 0.9524883089 | 0.9505397794 | 100 |
| CPB-v5.market-01 | 40 | 0.9708782664 | 0.9657628812 | 100 |

### past_relation

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history retention · **complexity 4/5** · saved TRAIN only. 112 natural-mask TRAIN rows; neural column averages the three saved fits. Coverage is complete.

| Method | Size | Linear head R² | Neural head R² | Coverage % |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01.init | 36 | 0.6514151377 | 0.8310905433 | 100 |
| CPB-v4.market-01 | 36 | 0.7543994049 | 0.9037237234 | 100 |
| CPB-v5.market-01.init | 40 | 0.6603006899 | 0.8974450571 | 100 |
| CPB-v5.market-01 | 40 | 0.7360101104 | 0.861776411 | 100 |

V5 has stronger saved TRAIN volatility readouts, while activity remains strongly
recoverable but slightly below V4. This is no activity-information collapse.
TRAIN fit quality cannot explain or predict generalization. Complete per-fit
MAE/RMSE, all three repetitions and the fixed early56/late56 results remain in
the machine diagnosis; physical units are logreturn, logreturn/tick,
log1p(base_volume) and correlation respectively. No head was refitted here.

## Reconstruction: all 112 and first 8 tell different stories

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history retention · **complexity 4/5** · saved TRAIN only. All112 ordered fixed-query TRAIN examples; equal-example/equal-channel/supported-cell reduction. NLL omits its constant and may be negative.

| Method | Rows | Standardized MAE | Gaussian NLL | Target cells |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01.init | 112 | 0.6641678566 | 0.4238153535 | 24192 |
| CPB-v4.market-01 | 112 | 0.2725080323 | -0.3566927721 | 24192 |
| CPB-v5.market-01.init | 112 | 0.6641678566 | 0.4238153535 | 24192 |
| CPB-v5.market-01 | 112 | 0.2544639357 | -0.3243773732 | 24192 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history retention · **complexity 4/5** · saved TRAIN only. Final saved TRAIN query subsets; first 8 are the only cellwise/gradient witnesses, not a representative replacement for all 112.

| Method | Subset | Standardized MAE | Gaussian NLL | Target cells |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01 | early56 | 0.290729209 | -0.2157708365 | 12096 |
| CPB-v4.market-01 | late56 | 0.2542868557 | -0.4976147077 | 12096 |
| CPB-v4.market-01 | first 8 | 0.2690123432 | -0.4834406341 | 1728 |
| CPB-v5.market-01 | early56 | 0.279320765 | -0.123670406 | 12096 |
| CPB-v5.market-01 | late56 | 0.2296071064 | -0.5250843405 | 12096 |
| CPB-v5.market-01 | first 8 | 0.2763843911 | -0.3937406086 | 1728 |

V5 has lower all 112 TRAIN MAE and lower MAE in both prespecified halves than V4,
yet its all 112 TRAIN NLL is less favorable. The first 8 witness instead has worse V5 MAE and NLL than V4.
Therefore its cellwise findings must stay local. The saved full-population
median and 95th-percentile MAE improve, while the 99th percentile and maximum
worsen; every row remains available in the diagnosis. This report
adds no TRAIN NLL advancement gate and uses no VALIDATION values to choose a loss.

## The surviving mean-weighting mechanism

For the saved NLL, the local mean derivative is error/variance. Detaching or
separating the variance-input representation does not remove that weighting.
The first 8 TRAIN witness shows that it places strong pressure on some small-error
price fields relative to larger-error volume/trade fields.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history retention · **complexity 4/5** · saved TRAIN only. Final first 8 TRAIN witness, 64 target cells in each shown channel/feature. These are per-cell absolute derivative summaries before the hierarchical averaging used in the local-adjoint table; no coefficient is fitted.

| Method / channel / feature | Mean absolute error | Mean inverse variance | Mean absolute NLL mean derivative | Mean absolute Huber1 mean derivative |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01 / C1 / F1 | 0.08819259127 | 23.32393127 | 1.573212507 | 0.08819259127 |
| CPB-v4.market-01 / C1 / F7 | 0.4669926616 | 2.962517154 | 0.8523573198 | 0.2900436395 |
| CPB-v5.market-01 / C1 / F1 | 0.07677476981 | 46.20263237 | 4.384018653 | 0.07677476981 |
| CPB-v5.market-01 / C1 / F7 | 0.4811928077 | 2.904815517 | 0.9498853369 | 0.3201005026 |

C1 is the canonical second market channel; F1 is high price and F7 taker-buy
base volume under the frozen nine-feature loader. This deterministic example
illustrates a broader pattern in the full 27-channel/feature record; it is not
an additional fitted selection or a claim about all 112 query cells.

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history retention · **complexity 4/5** · saved TRAIN only. Final first 8 masked-query natives; mean4 adjoints use the same basis and exact saved support/reduction. F64 scalar algebra approximates saved F32 intermediates, not gradient bit parity.

| Method | Mean-NLL adjoint norm | Huber1 adjoint norm | NLL–Huber1 cosine | NLL–MAE cosine |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01 | 0.01141879433 | 0.002169529514 | 0.7892655197 | 0.689971023 |
| CPB-v5.market-01 | 0.01210949467 | 0.002604280055 | 0.7080088179 | 0.6354359652 |

These directions are positively aligned, with differences in emphasis;
they are not evidence of opposing gradients. Initial mean4/native-variance
adjoints are zero because the corresponding decoder/projection starts at zero;
that is not a diagnosis of failed encoder learning.

V4's applied variance-to-mean4 derivative is exactly zero. Its 0.09374009403705622
saved-coefficient variance sensitivity is hypothetical and was not applied.
V5's live uncertainty4 adjoint norm is 0.04174171419695833, in a different basis
from mean4. It is scale dependent. No cosine, orthogonality, dominance or conflict
claim compares independent mean4/uncertainty4 coordinates. We omit upstream
bank-projection/parent32/encoder parameter Jacobians and optimizer replay.

## Where the saved readout gets its information

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history retention · **complexity 4/5** · saved TRAIN only. Final 112 natural-mask TRAIN rows; raw-coordinate population covariance trace and participation ratio. These describe coordinate scale/distribution, not intrinsic information or a learned post-encoder transform.

| Method | Block | Width | Covariance trace | Participation ratio |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01 | learned20 | 20 | 331.2684162 | 3.548824869 |
| CPB-v4.market-01 | fixed_spectral12 | 12 | 1.529628398 | 6.855212144 |
| CPB-v4.market-01 | mean4 | 4 | 14.60953303 | 3.219788006 |
| CPB-v5.market-01 | learned20 | 20 | 313.3400828 | 3.593198764 |
| CPB-v5.market-01 | fixed_spectral12 | 12 | 1.529628398 | 6.855212144 |
| CPB-v5.market-01 | mean4 | 4 | 9.608696907 | 3.078265623 |
| CPB-v5.market-01 | uncertainty4 | 4 | 87.54752256 | 1.033541219 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history retention · **complexity 4/5** · saved TRAIN only. past_volatility; already fitted ridge record 2701 decomposed into native blocks, with contribution SD in logreturn. This is neither a block ablation nor a refit; all saved repetitions are retained.

| Method | Block | Contribution SD | Contribution–target Pearson | Contribution–saved-residual Pearson |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01 | learned20 | 5.166180078E-05 | 0.03390743458 | -0.06381604319 |
| CPB-v4.market-01 | fixed_spectral12 | 4.78750153E-05 | 0.2229333928 | -0.00678580996 |
| CPB-v4.market-01 | mean4 | 0.000479668196 | 0.9664576768 | -0.04680385245 |
| CPB-v5.market-01 | learned20 | 4.753875402E-05 | -0.642636395 | -0.04801131919 |
| CPB-v5.market-01 | fixed_spectral12 | 2.983549E-05 | 0.2586198026 | -0.005569192218 |
| CPB-v5.market-01 | mean4 | 7.104871554E-05 | 0.765256163 | -0.01854571856 |
| CPB-v5.market-01 | uncertainty4 | 0.0004707502486 | 0.9853352663 | -0.01956135327 |

Dataset: **MKT-HIST-2** · BTC/ETH/SOL January 2024 minute-history retention · **complexity 4/5** · saved TRAIN only. past_activity; already fitted ridge record 2701 decomposed into native blocks, with contribution SD in log1p(base_volume). This is neither a block ablation nor a refit; all saved repetitions are retained.

| Method | Block | Contribution SD | Contribution–target Pearson | Contribution–saved-residual Pearson |
| --- | --- | --- | --- | --- |
| CPB-v4.market-01 | learned20 | 0.5016277215 | 0.9773826043 | -0.0352690221 |
| CPB-v4.market-01 | fixed_spectral12 | 0.05790186203 | 0.1391582142 | -0.009913959504 |
| CPB-v4.market-01 | mean4 | 0.1186400577 | 0.8739713239 | -0.01253935886 |
| CPB-v5.market-01 | learned20 | 0.4572315948 | 0.9677120907 | -0.03499756035 |
| CPB-v5.market-01 | fixed_spectral12 | 0.06138815851 | 0.2241818983 | -0.006774712518 |
| CPB-v5.market-01 | mean4 | 0.1793417397 | 0.9012560464 | -0.02668900589 |
| CPB-v5.market-01 | uncertainty4 | 0.04576661289 | -0.415620827 | -0.05068973581 |

The volatility predictor's main target-aligned contribution moves from V4 mean4
to V5 uncertainty4, whose TRAIN participation ratio is near 1. That supports
preserving the independent uncertainty branch as useful exported representation;
it does not make decoder variance identical to the volatility target or prove
that its scale/rank caused a score change. Raw covariance/adjoint scales cannot
rank the branches' usefulness.

Activity remains strongly carried by learned20 and mean4. Their saved scalar
ridge contributions have positive covariance/correlation in both versions
(V4 Pearson 0.8369762146875267, V5 Pearson 0.8590226333301162). This gives weaker
support to separately supervising parent/residual means as a cure for supposed
branch competition. Negative correlations of some small correction components
are valid readout decompositions, not proof of harmful training gradients.
Chronological component summaries and every row are retained without regime
selection. Fixed spectral statistics remain numerically identical.

## Recommended next controlled test, subject to review

Retain V5's separate mean4/uncertainty4 branches and native40 forward capacity.
Give the live mean route a fixed unweighted Huber(delta1) mean objective.
Calibrate variance with the existing Gaussian residual/logvariance objective
around a detached mean; keep the independent uncertainty projection and variance
head live. Keep support, hierarchical reduction, fixed delta 1 and positive link.
This replaces the remaining inverse-variance mean objective; it is not a tuned
blend, coefficient search, head rescue or extra-capacity experiment.

Keeping the global gradient-norm clip at 1 and one AdamW optimizer preserves
optimizer coupling. Zero
direct NLL mean gradient does not make optimization completely independent;
the mean residual changes the future variance-calibration targets as well.

That proposal tests one clearer hypothesis than another capacity change: mean
learning should no longer let predicted variance reweight reconstruction pressure,
while the useful exported uncertainty branch can still learn variance structure.
It does not optimize the complete joint-NLL mean gradient. Huber supervision may
underfit rare large errors or worsen NLL, volatility retention or timing; the
stronger prospective retention protocol must catch those failures.

Rank independent parent/residual mean supervision behind this test: current
saved ridge evidence shows complementary activity components, and the diagnosis
has no parent-mean/encoder Jacobian evidence that proves competing mean paths.
Do not widen, shrink, rescale or retune the stopped V5 recipe to recover a gate.

Any accepted next test must own a new complete named implementation and card,
start learned values and permitted TRAIN statistics/optimizer fresh, and declare
its geometry, fixed heads, seeds, budget, trust proofs and retention criteria
before execution. Never initialize from V3/V4/V5 or historical future-trained
weights. Preserve every checkpoint and scoped leader. No next version, model,
source/card or automatic dev promotion is declared in this report.

The separate [V5-retention draft](clean_causal_encoder_search_goal_v5_retention.md)
protects the new V5 records prospectively while leaving the original goal and
all old pass counts untouched. It adds no TRAIN NLL condition. Independent
confirmation on a separately frozen population remains necessary before any
generalization claim; it is not opened for mechanism selection here.
