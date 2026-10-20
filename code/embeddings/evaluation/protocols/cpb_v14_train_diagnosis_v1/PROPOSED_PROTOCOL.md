# Proposed V14 saved TRAIN diagnosis

18 October 2026. **Preparation only: not declared, admitted, implemented or executed.**
Execution requires completed V14 metadata preservation and a separate actual
declaration, exact input roles, source/binary identities, engineering fixtures and
independent reviews frozen before research payload access. This proposal grants
no execution, training, model choice or selection. Work states remain in Hero 17.

## Question and boundary

Describe native covariance and captured nested-window overlap, and decompose the
local shared-position adjoint at the two existing TRAIN captures. Reuse audited
V13 summaries as opaque context. These observations cannot identify the cause of
held-out losses or explain an uncaptured training trajectory. No validation/TEST
arrays, targets, predictions, head weights, checkpoints, refits, model/head
execution, finite differences, tuning or V15 selection are part of this proposal.

## Closed numerical inputs

Paths are current repository-relative locators. Preserve original
`cuwacunu_embedding` repository/path/bytes/SHA256 tuples through verified aliases.
Pins are transcribed from the saved Complete mirror. Draft preparation performs
no decoding or PT hashing; separate metadata verification receipts record any
fresh byte checks and are not replaced by this proposal.

| Role | Current location | Bytes | SHA256 |
| --- | --- | ---: | --- |
| Initial TRAIN native | `code/embeddings/output/runs/causal-fresh-v14/run-001/quality-001/features/initial/market/cpb_v14-train.pt` | 39403 | `2dd8ebd3b99ece38b7af4bd29103785d6b469820edcf5f1ab3b526ea62e133be` |
| Final TRAIN native | `code/embeddings/output/runs/causal-fresh-v14/run-001/quality-001/features/final/market/cpb_v14-train.pt` | 39403 | `9d6684b77f2ab0bfeec8d87a7fc1d1e52035c62a9183d1e64f68703c319c5f01` |
| Actual update 1 | `code/embeddings/output/runs/causal-fresh-v14/run-001/witnesses/training-update-1-components.pt` | 898684 | `6cd839de70f541969fcd04156421948513cecf5b83cfe4e40276651954161dd9` |
| Actual update 2048 | `code/embeddings/output/runs/causal-fresh-v14/run-001/witnesses/training-update-2048-components.pt` | 898885 | `404f69b16782494a05a6d1b2e6ba488612778038529fcc4c9f485e93180292e5` |

Historical feature paths use `/opt/cuwacunu/research/causal-fresh-v14/attempt-001/quality-001/features/`.
Historical witness paths use `/opt/cuwacunu/research/causal-fresh-v14/attempt-001/run-001/witnesses/`.
Suffixes are the exact feature/witness suffixes in the table. Current container
locators are `/mityane/` plus the complete current repository-relative paths.

Feature fields are `row_ids` Int64[112], `values` Float64[112,40], and `valid`
Bool[112]. Require all112 valid, finite values, and exact initial/final row-ID
association. Freeze row order from audited V13 half-row IDs and split context;
These IDs are exact query endpoint timestamps in epoch milliseconds, as used by
the frozen query plan and market loader. Preserve their original order and
identity. TRAIN spans1704067259999 through1705881599999 milliseconds.

The witness allowlist is32 of60 fields: `update`, `attempt`, `clip_factor`;
`temporal_export_weight_before`, `temporal_export_weight_clipped_gradient`,
`temporal_export_bias_clipped_gradient`, `temporal_pool_positions_clipped_gradient`;
`temporal_patch_positions`, `temporal_patch_valid`, `temporal_patch_rows`; and,
for each prefix `temporal_`/`temporal_contextual_`, `hidden`,
`pool_local_weights`, `pool_global_weights`, `pool_mixture_weights`,
`pool_bin_valid`, `pool_bin_mean`, `pool_bin_variance`,
`pool_bin_regularized_root`, `pool_bin_dispersion`,
`pool_projected_upstream_defined`, `pool_projected_upstream_pre_clip`.

Require the original card's exact dtypes/shapes: hidden[R,T,64], local/mixture
weights[R,4,T], global weights[R,T], bins Bool[R,4], moments/root/dispersion
Float32[R,4,64], upstream Float32[R,32], defined flag Long scalar, W[32,64],
bias[32], positions-gradient[16,1], and Float64 scalar clip factor in (0,1].
Packing positions are valid original0..15 or padding-1; packed rows are unique
batch/channel indices in0..23. Update2048 uses before-step2047 coefficients.

## Opaque V13 and metadata context

Under `.archive/embeddings/pre-cleanup-2026-10-18/doc/`, bind:

| Record | Bytes | SHA256 |
| --- | ---: | --- |
| `causal_v11_v13_train_diagnosis/diagnosis.json` | 4183840 | `716959d4428b3b65d71d4e0fa84fe5d96a84bdd9ab40266785b837d84f54ab27` |
| `causal_v11_v13_train_diagnosis/independent-audit.json` | 52464 | `0d47cc5306b6fbfae2a886c0cf1d1a90dc90a9b18c767cffef8357fdb20d3d11` |
| `causal_v11_v13_train_diagnosis/completion.json` | 5632 | `5b98ba9adeec76b5bb42aed6b398f1c82fd8d014aa903746c09401e620b4ba0f` |
| `CAUSAL_FRESH_V14_CARD.json` | 140239 | `546e3c15488169b0d416c6264159234c5840fdb71a9e0739599c922de57ca4e1` |
| `causal_fresh_v14/independent-training-audit.json` | 81346 | `43cf63c5251b0975e0e8c9d10950ea881e879b34a22b910fc7ee4b33648d50ee` |
| `causal_fresh_v14/complete-mirror-attempt-001.json` | 405883 | `24b6eacf71e4291a3880225c6e07669640dd4fde31497d8657667b3963cab7c9` |

Copy only `models.v13.stages.initial`, `models.v13.stages.final`,
`temporal_moment_observations.csv`, and
`temporal_moment_observations.actual_witnesses` as the four opaque reference
objects. Preserve their content and original labels. No V13/V11 tensor reads or
numerical replay; reused prior audit counts are not added to new check totals.
Bind the actual completed V14 closeout/alias/preservation receipts separately;
their paths, hashes and passed outcomes must not be invented in this draft.

## Fixed measurements

1. Native initial/final populations are full112, chronological early56 and late56.
   Blocks are learned/content20[0:20], spectral12[20:32], mean4[32:36],
   uncertainty4[36:40], complete40. For each coordinate retain mean, population
   SD, minimum and maximum. With n valid rows, use compensated finite sums:
   `mu_i=sum_r(x_ri)/n`, `C_ij=sum_r((x_ri-mu_i)*(x_rj-mu_j))/n`;
   retain `trace(C)`, `sum_ij(C_ij^2)`, and participation
   `trace(C)^2/sum_ij(C_ij^2)`. A zero denominator gives null. No epsilon,
   normalization, PCA, eigensolver or fitted transform. Missing actual112
   coverage is a refusal, not permission to silently shrink the population.
2. For each capture, route and packed row, let V be valid tokens and
   `S_j={t in V:16-L_j <= position_t < 16}`, L=[4,8,12,16]. Retain presence,
   local counts, pairwise intersection/union counts and Jaccard; empty union
   gives null. Assert exact nested membership and S_16=V. Use original positions
   rather than packed indices, last-visible positions or support counts.
3. Use captured local l, global g and mixture m without inversion/renormalization.
   For present windows retain weight mass, maximum, sum-squares and the prior
   V13 descriptor `1/sum(m_t^2)`. Pairwise local/mixture overlaps are
   `sum_t min(l_it,l_jt)` and `sum_t min(m_it,m_jt)`; retain mixture L1 distance.
   If either window is absent, weighted pair descriptors are null. Summarize
   present-row distributions with fixed quantiles0,.01,.05,.25,.5,.75,.95,.99,1,
   linearly interpolating at `(n-1)*q`; retain actual denominators and raw vectors.
   Longest-local/global equality and the shared global half are structural
   properties; they do not establish representational collapse.

## Bounded shared-position adjoint

For one present row/window/route use saved hidden x, mean mu, root r, mixture m,
PRECLIP upstream and its before-step W. Within window j define
`u_d = saved_upstream[row,8*j+d]` for d=0..7. Indices d=0..3 select the mean
half; d+4 selects dispersion. Every sum below is compensated finite F64 arithmetic.

```text
a_k = sum_(d=0..3) u_d * W_(8j+d,k)
b_k = sum_(d=0..3) u_(d+4) * W_(8j+d+4,k)
delta_tk = x_tk - mu_k; epsilon_k = sum_valid_t m_t * delta_tk
beta_k = b_k / (2 * saved_root_k)
qMean_t = sum_k a_k*x_tk
qDisp_t = sum_k beta_k*(delta_tk^2 - 2*epsilon_k*x_tk)
dsLocal_t(q) = 0.5*l_t*(q_t - sum_s l_s*q_s)
dsGlobal_t(q) = 0.5*g_t*(q_t - sum_s g_s*q_s)
G_position[p] = clip_factor * sum_routes,rows,windows,moments,branches,t:position_t=p ds_t
```

Retain route/window/mean-or-dispersion/local-or-global length16 components and
their sum. They are analytic reference components, not separately captured
autograd gradients. Compare the summed clipped vector to the saved position
gradient. Also verify export W/bias: accumulate `u_d*saved_mean_k` or
`u_(d+4)*saved_dispersion_k`, and u for bias, then clip once after the full sum.
Retain component norms, pairwise cosines and `norm(sum G)/sum norm(G)`;
retain a zero component norm as0. A pairwise cosine is null when either norm
is0; the cancellation ratio is null only when its denominator is0. Exact
cancellation of nonzero components gives ratio0. These describe local
cancellation in two captured batches, not global gradient conflict, clipping
damage or task importance.

## Numerical/refusal contract and implementation boundary

- Retain the original two-sum mean. Check m=0.5*l+0.5*g for present windows,
  zero otherwise; weight storage outside declared support is exact zero.
- Mask unsupported hidden operands before arithmetic. Require finite supported
  values; zero-times-NaN is not masking. Absent windows contribute no adjoint;
  check raw mean/variance0, raw root1/32 and dispersion0. The served whole8
  slice is zero by frozen source construction; the32-field allowlist does not
  include the forward projected values or before-step export bias needed to
  verify that served slice directly from arrays.
- Use the actual saved F32 root. Validate F32 add→sqrt→subtract with tau2^-10
  and offset1/32. Tiny dispersion rounded to0 does not null its derivative.
  Keep the finite-F32 center residual epsilon and both attention branches live.
- Explicit undefined-route flags require exact zero fallback. Existing captures
  mark ordinary upstream undefined and contextual defined; no cross-route
  competition claim follows. Sum PRECLIP components, then apply clip exactly once.
- Saved F32 moment checks: abs(error)<=3e-5+3e-5*abs(reference). Local W/bias/
  position parity: abs(error)<=1e-9+3e-4*abs(reference); each capture's expected
  postclip export maximum must exceed1e-6. F64 independent descriptor/algebra
  checks retain1e-10+1e-9*abs(reference). Masks/flags/IDs/counts and required zeros
  are exact. Common score-bias gauge remains descriptive, without a parity gate.
- Refuse changed pins, source/card mismatch, unexpected fields/types/shapes,
  invalid packing, nonfinite supported arithmetic, overflow or failed parity;
  preserve exclusive failed/partial output without retry or replacement.
- Smallest new adapter: C++ CPU LibTorch `InputArchive`, explicit field/schema
  access, ordinary vectors and independent `finite_sum`/F32 helpers; no encoder,
  decoder, head, optimizer or autodiff implementation. LibTorch may deserialize
  the whole admitted TRAIN archive: selected fields used is not a guarantee of
  selective storage decoding. A stricter reader needs separately reviewed
  actual-artifact compatibility; no automatic reader/Python fallback.
- Use the unchanged `/opt/cuwacunu_embedding/libtorch` SDK, ABI1, and
  `-fno-fast-math -ffp-contract=off`. New code/JSON have new source identities.
  Build through root tasks in a named `/opt/mityane-geometrics/embeddings/build/`
  directory; preserve selected new results under ignored `.temp/embeddings/runs/`.
  Freeze the actual source closure, binary, role ledger and engineering evidence,
  then independently review/audit. Hash inputs before/after; copy opaque V13
  objects exactly; disclose arithmetic counters without adding reused audits.

Frozen historical sources (`cuwacunu_embedding`); never substitute edited source:

| Original path | Bytes | SHA256 |
| --- | ---: | --- |
| `code/encoders/causal_patch_bottleneck_mae/versions/v14/include/embedding/encoders/causal_patch_bottleneck_mae/detail/v14/temporal_pooling.h` | 9964 | `1539d0e6684c39c8ae5fc6d461a0c7bf3ab836b8643d9deb97a2a7d2d0006785` |
| `code/research/causal_fresh_v14/train_main.cpp` | 64653 | `3362c11a659bb57242ac7d2295ea71ab18fe7f687f85aa06ff3fed744d7c56b1` |
| `/opt/cuwacunu/research/causal-fresh-v14-training-audit/attempt-001/source/audit_training.py` | 149059 | `b2d9cf30780dcc40270474692d479f5f6190b318663f08af4ac6c3716fcf5467` |

No full112 temporal captures, hidden adjoints, other2046 raw-gradient archives or
matched pre-step decoder replay exist. This draft creates no result or work-state change.
