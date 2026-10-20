# Synthetic response study: implementation interface

This is the internal interface for `OR06-SYNTH-RESP-001`, version `1.0.0`.
The prospective protocol and instance RER govern scientific interpretation.

## Generator and response

All raw coordinates are integer numerators over 4096. Baseline `a_num` is
uniformly drawn from integers [-3072,3072]; `b_num` from [-4096,4096]. Within
each seed, reserve a unique `b_num` for each source unit across all splits, so
no alternative (which holds b fixed) duplicates another split's inputs.

An unsigned 32-bit LCG advances as `s=(1664525*s+1013904223) mod 2^32`.
For each unit draw `a_num=(next mod 6145)-3072`, `b_num=(next mod 8193)-4096`;
retry the whole draw if b was already used. Draw noise sign from the next
state's high bit (one means +1, zero means -1). Split order is training,
development, assessment; IDs are `<split>-<zero-based-index>`.

Synthetic reference map is `F(a,b)=1+2a-b+1.5ab+0.5a^2`.
Training labels are `F(a,b)+noise_sign/32`. Development/assessment reference
contrasts use noiseless F. Fixed changes to a have numerators
`[-1024,-512,0,512,1024,8192]`; the last is a support stress alternative.
All supported source attributes lie in [-1,1]. b is held fixed.

Seeds are 1103, 2207, 3301. Per seed: 512 training, 128 development and
256 assessment units. No hyperparameter selection occurs. Study assessment
is generated only after model sealing. Separate engineering self-check seeds
and small populations may be used before freezing; they are not study evidence.

## Four independently fitted response arms

An intercept plus train-standardized features uses ordinary least squares,
normal equations with partial pivoting, pivot refusal threshold 1e-12, with no
regularization or search. Standardization uses training population mean and
population standard deviation (`sqrt(sum((z-mean)^2)/n)`), not assessment data.

| Arm | Fixed feature vector | Head parameters |
| --- | --- | --- |
| `mixed` | `[a+b,a-b,ab+a^2,ab-a^2]` | 5 |
| `raw_polynomial` | `[a,b,ab,a^2]` | 5 |
| `raw_linear` | `[a,b]` | 3 |
| `lossy` | `[b,b^2,b^3,b^4]` | 5 |

The matched-capacity raw-polynomial arm has equivalent expressive power to
mixed; no superiority claim between them is expected. The zero-response
baseline always outputs delta zero and needs no fitted head.

## Native command and records

`synthetic_response fit TRAIN.tsv MODEL.tsv SEED`
and `synthetic_response evaluate INPUT.tsv MODEL.tsv OUT_DIR SEED SOURCE_SHA CONFIG_SHA`.
Paths are explicit. Output files are created exclusively; existing files are
preserved. The existing shared std-only SHA256 implementation is reused.

Input TSV header is `unit\ta_num\tb_num\tnoise_sign`, followed by exact integer
values. Rows must be unique and finite within generator bounds; noise is +/-1.
`fit` accepts training unit IDs only. `evaluate` accepts development or assessment
unit IDs, never a mixture. No evaluator fits or updates any model.

Model TSV header is `arm\tfeature\tmean\tscale\tcoefficient`. Each arm has one
`intercept` row (mean 0, scale 1) then features `0` through `n-1`. All doubles
use round-trip precision (17 significant decimal digits), finite scales >0.

Output `pairs.tsv` header is:

```text
arm unit change_num a_num b_num alt_a_num state reason recorded_base recorded_alt delta z_base z_alt
```

Columns are tab-separated. `z_base` and `z_alt` contain comma-separated **raw**
feature coordinates in the arm's declared order. Recorded responses are from
the actual frozen model. `delta` is the operation's optional output, written
`NA` for every refusal. `reason` uses the public reason code, or `NONE` on
success. Each unit has six rows per arm, in configured arm and change order.

Also retain actual provider/config/model/space/pipeline descriptors and a
content-addressed `artifacts.tsv` store with header `sha256\tpayload_hex`.
Every non-source ArtifactRef in the operation request resolves either to a
retained descriptor file or an exact payload in this store. Response payloads
include their numerical values; input payloads identify raw coordinates;
representation payloads include actual feature coordinates. Shared source refs
identify the input TSV row. Hash all actual payloads; do not invent digest pins.

Evaluate uses non-temporal mode with explicit synthetic-static justification,
known deterministic pipeline revision, exact input-role/space, output position
0, mask-valid evidence, and constructor/support/control policies bound to the
saved source/configuration. Alternative origin is edited_input and candidate
kind stays transformed_candidate. Roster and per-pair counts must be retained;
no refused pair is silently excluded. A support stress row abstains because
its a value exceeds [-1,1]. No timing/causal interpretation is claimed.

## Independent audit and frozen metrics

The auditor reconstructs models from saved coefficients, verifies train-only
scalers/features/response records and artifact bytes, and independently derives
both model contrasts and the closed-form reference
`delta_F=h*(2+1.5b+a)+0.5h^2`. Numerical audit tolerance is 1e-10 absolute.
For each seed/arm, primary task error is the mean over source-unit MAEs across
the five in-support changes, including the no-change control. Report zero
baseline, each arm, no-change maximum error, all denominators and stress states.

Positive benchmark screen (all required, no best-seed selection): mixed MAE
<=0.02 in each seed; mixed/raw-polynomial MAE difference <=1e-9; pooled mixed
MAE <=20% of pooled zero-baseline MAE; in-support coverage 100%; stress abstention
100%; numerical independent audit passes. A failed screen can still be a valid
negative result. Contrasts versus F include response-model fit error and do
not diagnose operation arithmetic in isolation.

Uncertainty: source-unit bootstrap within each assessment seed, preserving all
five alternatives; 2,000 resamples with LCG seed 9001, macro-average the three
seed MAEs, sort, and use indices floor(0.025*2000)=50 and
floor(0.975*2000)=1950. This is conditional bootstrap sensitivity for the fitted
models; per-seed results separately describe training-seed variability. No
real-world or causal generalization follows from this synthetic benchmark.
Draw order is resample, then configured seed, then source-unit slot. Each index
is `floor(next_LCG / 2^32 * unit_count)`, with common draws across arms. Do not
use the low-bit modulo index, which degenerates at a 256-unit roster.
