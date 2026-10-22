# Export RPB with its complete training method

This package delivers configurable fresh RPB-v22, its v20/shared numerical
dependency closure, streamed TRAIN fitting, masks/objective/update method,
typed continuation, native example, configurations, fixtures, research context
and licenses. A trained checkpoint is a separate instance. No reference weights
are needed for fresh construction or the artificial example.

The [inventory profile](package.json) curates source entry files and bounds the
recursive quoted-include closure. The [method](TRAINING_METHOD.md),
[requirements](DEPENDENCIES.md) and [compatibility contract](COMPATIBILITY.md)
travel with the code. Copying only the new `family/` headers omits the preserved
numerical implementation and is incomplete.

| Included source | Purpose |
| --- | --- |
| Family headers and preserved numerical closure/wrappers | Configurable construction/discovery, preprocessing, encode, real update/skip, inference and typed save/resume |
| Native trainer, generic observation archive/writer and fixtures | Runnable artificial fresh/fit/train/save/resume/export workflow independent of market data |
| Configuration examples and focused tests | Meaningful single/multi-channel geometry/capacity/output examples and explicit refusals/continuation |
| Complete learning method and historical records | Numerical rules, portable versus historical recipes, research rationale, failures and temporal/ancestry limits |
| Licenses, provenance, protocol and portability receipt | Origin attribution and actual measured engineering scope |
| Exporter/verifier and test, source profile and generated manifest | Repeatable mapping, dependency admission, per-member SHA256 and offline verification |

## Export and independently verify

Run from Mityane's repository root in the existing managed environment. Choose
a new output leaf under ignored `.temp/`; create its parent separately when
needed. No dependency installation, SDK copy or container lifecycle change is
part of export.

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/node.sh',
  'code/embeddings/scripts/export-rpb-family.mjs',
  '--output', '/mityane/.temp/embeddings/exports/rpb-family-2026-10-22'
)
```

Retain the printed **manifest SHA256 independently** with the delivery. Its
`source_closure_sha256` and per-file paths/hashes bind the selected working-tree
bytes. Git HEAD records ancestry; it does not claim local edits were committed.
The exporter admits sources and dependencies before creating the output leaf,
refuses existing output, unsafe paths, symlinks, artifacts and missing/ambiguous
or out-of-scope quoted dependencies. Failed output remains for inspection.

```powershell
.\container.ps1 -Action exec -Command @(
  'bash', 'code/scripts/node.sh',
  'code/embeddings/scripts/export-rpb-family.mjs',
  '--verify', '/mityane/.temp/embeddings/exports/rpb-family-2026-10-22',
  '--expected-manifest-sha256', 'REPLACE_WITH_INDEPENDENTLY_RETAINED_SHA256'
)
```

After relocation, a recipient can run its own available Node against
`sources/code/embeddings/scripts/export-rpb-family.mjs --verify BUNDLE
--expected-manifest-sha256 PIN`. Verification reads only bundle members. It
checks the independent pin, every member's bytes/role/path mapping, the curated
profile, recursive local-include closure and complete bundle inventory. It
requires no origin or sibling source checkout. Node is copying/metadata tooling;
numerical training/inference remain C++.

## Build and accept an import

Preserve repository-relative files under `sources/`. Add the five declared local
include roots and both Torch include roots to the recipient's managed Linux
root build, with C++20, ABI1 and the recorded libraries. `family.mk` provides
opt-in targets and accepts `RPB_FAMILY_SOURCE_ROOT=BUNDLE/sources`; it does not
replace the consumer's environment or root build. A consumer may call the
native executables directly instead of Mityane's Node runner.

A minimal recipient acceptance Makefile can use the delivered target fragment.
Save this harness in a named container build directory and set its paths to
that recipient's existing SDK, bundle and approved build/output locations:

```make
CXX := g++
LIBTORCH := /opt/cuwacunu_embedding/libtorch
RPB_FAMILY_SOURCE_ROOT := /absolute/path/to/BUNDLE/sources
BUILD_DIR := /opt/mityane-geometrics/embeddings/build/hx16-rpb-import
RUN_ROOT := /mityane/.temp/embeddings/runs/hx16-rpb-import
CXXFLAGS := -std=c++20 -O1 -g0 -Wall -Wextra -MMD -MP
LDFLAGS := -L$(LIBTORCH)/lib -Wl,-rpath,$(LIBTORCH)/lib -Wl,-rpath-link,$(LIBTORCH)/lib
LDLIBS := -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10 -Wl,--as-needed -pthread
include $(RPB_FAMILY_SOURCE_ROOT)/code/embeddings/encoders/raw_patch_bottleneck_mae/family/family.mk
```

Inside that managed environment run `make -f /absolute/path/to/recipient.mk
build-rpb-family build-rpb-family-trainer build-rpb-family-fixtures`. This
compiles exclusively delivered source and the recipient's SDK; Mityane's root
Makefile is not in the bundle. The native fixture takes three positional
arguments: `test_rpb_family SINGLE.CONF MULTI.CONF NEWLEAF`. Use the delivered
`family/config/single.conf` and `family/config/multi.conf` paths beneath
`BUNDLE/sources/code/embeddings/encoders/raw_patch_bottleneck_mae/` and the
resulting binary. For the complete CLI fixture,
use the recipient's available Node to run delivered `family/tests/training_example.mjs`
with `--trainer BIN --fixture BIN --variant BIN --config-directory DIR --output
NEWLEAF`. The native CLI and [training guide](../examples/TRAINING.md) also work
without Node orchestration. Build/tmp paths stay inside the container; selected
reports and exports use separately named ignored artifact directories. These
commands build/check already configured source, and do not create containers
or install toolchains.

Compile and exercise the imported native fixtures/example under the consumer's
own pinned SDK and record source/config/runtime/binary/output identities. Byte
verification and original engineering receipts do not certify a relocated
build. The [bounded protocol](../evidence/PROTOCOL_2026-10-22.md) and
`../evidence/portability-2026-10-22.json` bind producer engineering coverage.
Export and isolated acceptance are recorded separately in
`delivery-2026-10-22.json` beside this guide. That receipt is kept outside its
own exported manifest to avoid a circular hash binding. Consumers check the
actual receipt and retained manifest pin; original-path proof does not certify
an import or admit application data.

Each recipient records its retained delivery pin, complete original/destination
mapping, any adaptation, runtime and acceptance. Origin paths are evidence
identities; relocation is a new path-bound source instance. The first portable
delivery does not select an application checkpoint or change Cuwacunu.

## Producer and application ownership

Mityane owns reusable fitting arithmetic, masking/target support, loss and
gradient routes, optimization primitives, stochastic/progress rules, raw
formats, bounded fixtures and portability evidence. Applications own real
semantics/populations, admitted TRAIN/calibration membership, complete recipe
choices and substantial training/scaling/recovery. They also own trusted
artifact selection/publication/capture, exact fitted descendants and downstream
utility. The [shared ownership contract](../../../COMPONENT_DELIVERY.md)
preserves this separation.

The example provides one explicit counter-driven modulo row sampler, original
patch masks and no extra context deletion. An application may choose a supported
recipe or name a changed mechanism and establish its evidence. Neither shared
codecs nor equal dimensions confer trajectory/coordinate compatibility.

Datasets, fitted states, weights, optimizer archives, generated runs, credentials,
SDKs and images are excluded. Historical v20/v22 research, native36's balanced
success and v30's market successes/transfer gap retain their original scope.
Noncausal-training ancestry limits remain visible. Frozen records are copied
as context, not rewritten or promoted to new path-bound evidence. This source
delivery creates no market, transfer, forecasting or application-quality claim.
