# Generic multiband screen v1

This separate development protocol compares new **RPB-v18.alt-01** instances
with **RPB-v19**, the mixed spectral-relation design. Existing producers,
checkpoints, instance bundles and defaults stay preserved.

Dataset **TEMPO-4**, designed complexity **5/5**, tests slow-component lead/lag
in a coherent two-rhythm mixture. Separate **AMP-2**, complexity **5/5**, tests
relative slow/fast strength. The unchanged generator and observed-information
admission are in [two-component information v1](../two_component_information_v1/README.md).
That data-only diagnostic never supplies encoder features or training targets.

The [prospective card](../../cards/multiband_screen_v1.md) fixes all architecture,
data, training, head and continuation rules. Two new cohorts per task are
generated once, retained as 12 typed legal CPU archives, and shared by the two
candidates. Each candidate retains its own initial0/trained512 controls, heads,
queries, full losses and CUDA checkpoints. Every encoder operation runs on CUDA.
CPU handles legal data, fixed heads and saved-only arithmetic checks.

Within the already running documented container:

```sh
bash code/evaluation/protocols/multiband_screen_v1/admit.sh
# Use the new admission path printed by that command, exactly once:
bash code/evaluation/protocols/multiband_screen_v1/run_screen.sh /embedding/output/runs/rpb-multiband-screen/admission-XXXXXX
python3 -B code/evaluation/protocols/multiband_screen_v1/publish.py --admission /embedding/output/runs/rpb-multiband-screen/admission-XXXXXX
```

Admission freezes SOURCE/card/SDK before engineering, builds in the named
container session, and tests CUDA arithmetic, updates, serialization and the
typed loader. The quality runner verifies those receipts and SOURCE, creates
one absent data matrix, freezes its bytes, then runs the two candidates
sequentially. It rejects existing output leaves. A failure preserves evidence.
The standard-library checker replays saved fits and queries; it cannot train,
fit heads, execute an encoder or repeat information-frequency search.

Publication groups by task and view, keeps initial controls and every cohort,
and reports the frozen joint continuation gate. No TEST, stress, selection,
post-encoder PCA or automatic promotion is authorized. A failed gate stops
this recipe's expansion. Full capsules are local ignored output; compact
source, reports and provenance are committed.
