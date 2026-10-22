# V17 January observation-window preparation

This model-independent C++ entry materializes the exact data binding in the
[prospective V17 protocol](../../../../embeddings/evaluation/protocols/cpb_v17_balance_v1/PROTOCOL.md).
It reads the three already admitted January2024 CACHE88 sources and writes the
[generic observation-window archive](../../../observations/OBSERVATION_WINDOWS.md).
It performs no fitting, encoding, target construction, optimizer update or
scientific scoring. Torch is used only for typed CPU tensor archives.

The root build owns its target and environment; use the existing managed Linux
container and `container.ps1`/`code/scripts/task.sh`. No component-owned build or
environment configuration and no Python is introduced.

```text
cpb_v17_windows --mode diagnostic|train|probes
  --cache BTC_PATH --cache ETH_PATH --cache SOL_PATH
  --cache-sha256 BTC_SHA --cache-sha256 ETH_SHA --cache-sha256 SOL_SHA
  --output NEWLEAF
```

An existing output is refused. The leaf's parent must exist. Exactly three
distinct source files and the original ordered protocol hashes are mandatory;
path aliases cannot change asset order. The helper verifies file sizes/hashes
before and after, hashes the actually loaded buffers, and checks the original
minute lattice using close-time metadata. Observation values are decoded only
inside the selected retained windows. Nonfinite/sentinel observations preserve
the original missing-support policy; supported F64 values must remain finite
under the declared historical F32 parent cast.

| Mode | Output | Selection |
| --- | --- | --- |
| `diagnostic` | `diagnostic-TRAIN.pt` | 64 evenly spaced legal TRAIN H128 anchors |
| `train` | `calibration-TRAIN.pt` | 1024 evenly spaced legal TRAIN H128 anchors |
| `probes` | `probe-TRAIN.pt`, `probe-VALIDATION.pt` | Original deterministic112/32 H256 nonoverlapping anchor geometry, retaining recent128 values |

All archives contain genuine CPU F64 `[B,3,128,9]` observations, Boolean support,
semantic IDs0/1/2, per-row original timestamps, cutoff/row IDs, minute cadence,
source/split identities and actual TRAIN/VALIDATION role. Source record F64 bits
and all serialized tensor bytes are checked on reload. `preparation.json`
records inputs, executable, rows/bounds, source schema, observed/genuine-F64 cell
counts and every output's size/SHA256. Failure leaves any partial leaf intact.

The `probes` mode is invoked by the study orchestrator only after all six final
instances are sealed. The data helper does not itself authenticate that seal.
Probe labels and original legacy-F32 target arithmetic are owned by the later
readout phase; raw F64 observation preparation does not construct those labels.

The reader independently owns a small CACHE88 layout and timestamp-selection
implementation derived from the preserved V16 study adapter. It includes no
encoder or historical model-binding header. Its only project dependencies are
the generic observation archive and shared artifact SHA256 header; see
[source provenance](provenance.json). Data/output files stay under ignored
`.temp/`; the original sources remain unchanged.
