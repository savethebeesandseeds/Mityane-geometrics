# Saved-coordinate usefulness diagnostic

This separate CPU evaluator consumes preserved CUDA exports of RPB-v20.alt-01
on TEMPO-5. It fits the unchanged shared classifiers to fixed12 once and to
learned20 at points0/512, reusing original complete32 and raw baseline results.
It constructs no encoder and changes no encoder weights or native export.

The [prospective card](../../cards/configurable_branch_v1.md) fixes the inputs,
27 new paired head fits, repetitions and reporting limits before execution.
All source cohorts remain known development evidence. Different branch widths
give different classifier parameter counts; these are sufficiency diagnostics,
not exact causal contributions or a new embedding version.

Use the existing managed `cuwacunu_embedding` container:

```sh
source /etc/profile.d/embedding.sh
bash code/embeddings/evaluation/protocols/configurable_branch_v1/run_branches.sh
```

The launcher uses a separately named internal build session, creates one unique
capsule, captures its source/card/binary and original-evidence fingerprints,
runs small engineering regressions, then fits only the declared branch heads.
It audits the persisted classifier arithmetic and independently publishes the
agreed tables. Original heads, quality data, checkpoints and reports are retained.

Reusable encoder/config files remain byte-identical with `../cuwacunu_dev`. This
diagnostic belongs to the separate evaluation repository and needs no market
loader or application dependency.
