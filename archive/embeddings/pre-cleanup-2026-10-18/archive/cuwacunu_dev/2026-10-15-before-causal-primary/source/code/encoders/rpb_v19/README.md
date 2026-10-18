# RPB-v19 encoder

This is the single supported encoder in the clean Cuwacunu integration project.
Its unchanged source comes from `cuwacunu_embedding` revision
`191cbaa44b921c37f8f458d474620e57a7924b0c`. RPB-v19.alt-01 is the fresh
confirmation of this same design, not a different architecture.

The public header is `include/cuwacunu/encoder/encoder.h`. The 17 headers under
`include/embedding/` are a byte-identical dependency closure. Their original
namespace, parameter names, inheritance, arithmetic and checkpoint loading stay
intact. Historical base classes inside that closure are implementation details;
this project exposes only RPB-v19. Evaluation and market ingestion remain outside
the encoder.

## Input and output

| Field | Contract |
| --- | --- |
| Input values | `[B,3,32,3]`: batch, channels, historical steps, features |
| Observed mask | Boolean, same shape; false values are unavailable |
| Channel IDs | int64 `[3]` or `[B,3]`; identify series independently of storage order |
| Endpoints | Optional finite floating `[B]` for direct calls |
| Time | One aligned uniform sampling interval |
| Native embedding | CUDA float32 `[B,32]`: learned shape20 plus fixed spectral12 |
| Reconstruction | CUDA float32 `[B,3,32,3]`, decoded from that sole native32 |

The batch dimension varies. Other dimensions are fixed for this design. Larger
histories, nine kline features or additional channels require a separately adapted
encoder and validation; changing configuration does not generalize the saved v19
checkpoint. Missing values remain masked, and entirely absent windows return an
invalid exact-zero native vector. Masking does not make irregular timestamps
equivalent to uniformly sampled input.

## Public API

```cpp
#include "cuwacunu/encoder/encoder.h"
namespace enc = cuwacunu::encoder;
auto config = enc::fixed_config();                 // CUDA:0, IDs 0/1/2, interval 1
enc::Encoder model(config);
auto scaler = enc::fit_train_scaler(training, config);
auto batch = scaler.transform(raw_batch, config);
model->eval();
torch::NoGradGuard no_grad;
auto encoded = model->encode(batch);
auto embedding = enc::native_embedding(encoded, config);
auto reconstruction = model->decode(embedding, batch.channel_ids);
```

`Config`, `Input`, `FrozenScaler`, `EncodeOutput` and `ForwardOutput` alias the
preserved implementation types. `Encoder` aliases the v19 model holder.
`fixed_config(device, channel_ids, sampling_interval)` constructs the closed
geometry with the required early channel mixer and global bottleneck. Model
execution requires CUDA and float32.

Fit the scaler on TRAIN only, then keep it unchanged for validation and serving.
`fit_train_scaler` currently accepts the whole training tensor, so it is not yet
suitable as a streaming fit over a massive dataset. `FrozenScaler::transform`
normalizes each selected batch on CPU and transfers it to CUDA; that is
preprocessing, not a second CPU model run. The old loader's causal log-ratio/log1p
normalization is a different transform. Application adapters must explicitly
choose their preprocessing instead of silently applying both.

`load_confirmation(path, config)` verifies the research artifact kind, candidate,
protocol, architecture, layout, training recipe, parameter counts and literal
frozen prior, then loads its associated model/scaler for inference. That format
requires original IDs 0/1/2, interval 1 and numerical settings. It does not import
the research optimizer, resume training or claim a market-ready checkpoint.
Verify SHA-256 against the reference manifest separately. Custom market metadata
needs a new explicit artifact contract rather than relabelling old confirmation.

## Focused checks

Compile `tests/import_test.cpp` with `-std=c++20`, this module's `include` directory,
the pinned LibTorch headers/libraries, C++11 ABI1 and pthread. No shared evaluator
or source `.cpp` files are needed.

Run the test in the managed Linux development container:

```text
rpb_v19_import_test
rpb_v19_import_test CHECKPOINT CONTROLLED_INPUT SAVED_FEATURES
```

The first route checks masks, variable batch sizes, rejection of shape changes,
model/scaler checkpoint parity and one disposable CUDA optimizer update that
must preserve all 432 frozen values. This is an import correctness fixture,
not a new research comparison. The optional route loads an existing confirmed
checkpoint and exactly compares imported CUDA inference with saved native32 and
validity. It performs no optimizer updates or execution of the sibling encoder.

Reusable changes to input/output dimensions, model arithmetic, masking, scaling
or serialization should be recorded and brought back to `cuwacunu_embedding`
under the policy in `doc/ENCODER_BACKPORT.md`. Dataloader and application adapters
stay in this integration repository.
