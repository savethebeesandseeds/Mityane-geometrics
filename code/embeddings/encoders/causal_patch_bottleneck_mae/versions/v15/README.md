# CPB-v15: matched V4 native36 objective study

Source preparation only. No successful build, engineering result, training,
quality result, application selection or encoder superiority is established by
these files. The actual two-arm, three-paired-seed study and chronological
confirmation require their separately frozen protocol and execution admission.

V15 owns a complete physical copy of the frozen V4 architecture closure under
`embedding::encoders::causal_patch_bottleneck_mae::v15`. Its public header is
`embedding/encoders/causal_patch_bottleneck_mae/versions/v15.h`. Runtime code uses
only its own headers, LibTorch and the C++ standard library. Copied source notices
remain in place. The source-copy record distinguishes original historical tuple,
archived locator, maintained V4 locator and newly owned V15 identity.

The one scientific change is the training objective. Unit Huber(delta1) supervises
the live parent32/mean-change4 path. Unit Gaussian variance NLL uses detached
mean error, while the conditional variance head still reads `mean4.detach()` as
in V4. Consequently variance calibration directly trains only its projection
and bias; it does not train native coordinates or the mean decoder. Reporting
Gaussian NLL is a separate detached copy and is never the backward tensor.
`trace.output.loss` and `trace.objective.total_loss` are the explicit composite.

The single shared global norm2 clip1, one AdamW, future residual coupling and
TRAIN-only statistics remain. This is not optimizer independence. The variance
forward formula, pooling, all module registration/constructor draws and native36
geometry are unchanged: learned20 + literal spectral12 + mean-change4. There are
75 learned tensors/293672 learned values and 3888 fixed prior values, 297560 total.
No independent uncertainty4, teacher, trained weight, optimizer or fitted
historical statistic is imported. `fresh(config, scaler, bank, seed)` remains the
sole initializer; a study supplies each actual seed explicitly.

Public objective metadata is `kObjectivePolicy`, `kGradientPolicy`,
`kDirectGradientRoutes`, `kMeanHuberDelta`, `kMeanLossWeight` and
`kVarianceLossWeight`. Both outer and nested typed inference/resume envelopes bind
the exact policy, delta1, unit coefficients, detached-error/input flags, direct
routes and reporting-only NLL flag. A V4 checkpoint is not a V15 checkpoint.

`v15.mk` defines build entries for three engineering fixtures. Root tasks own
actual invocation and all build artifacts stay under the named container build
directory. No standalone environment or lifecycle configuration is introduced.
The full fixture target requires explicit `CPB_V15_ENGINEERING_OUTPUT` (a new
leaf) and `CPB_V15_ENGINEERING_SEED`; the source does not allocate a study output.

- `v15_test.cpp` adapts the inherited full CUDA causal/typed-resume fixture to the
  composite objective and adds objective-field refusal cases.
- `v15_variance_gradient_test.cpp` preserves the V4 forward/variance-input-stop
  compatibility proof. Its explicitly called Gaussian helper is a positive
  control; its mean gradient is not the V15 training gradient.
- `v15_objective_test.cpp` checks independent masked F64 values/derivatives,
  hierarchical reductions, detached variance error/input, live variance-head
  learning, invariance of direct Huber mean gradients to variance, unsupported
  NaN sanitization, malformed supported values and the all-empty case. It also
  pairs every initial parameter/buffer and actual native/decoder forward with
  a fresh same-seed V4 control, verifies capacity and applies one shared clip to
  a synthetic combined-gradient stress. No optimizer step or historical
  checkpoint is used. The V4 comparison headers are engineering-only dependencies
  with separately admitted current path identities.

The mechanism fixture accepts `--settings`, `--control-settings` and `--seed`.
Its default 945101 is an engineering seed, not an admitted training-seed list.
New-period readout definitions, sealed data roles, TRAIN statistics, all fixed
heads and protection floors belong to the actual study protocol. No parameter
or condition is tuned from engineering or the repeatedly inspected validation.
