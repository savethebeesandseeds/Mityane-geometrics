<!-- Copyright (c) 2026 Waajacu. All rights reserved. See ../../../../LICENSE. -->

# Causal encoder application version catalog

The [live index](index.json) selects only **CPB-v3.market-01** for cuwacunu_dev.
[CPB-v3.json](CPB-v3.json) describes the copied design and
[the dev instance record](instances/CPB-v3.market-01.json) binds the selected
checkpoint, provenance and authoritative research evidence. This catalog is
application metadata; the original prospective research card and frozen records
remain unchanged in the research repository.

The sole active physical implementation is [`v3/`](v3/). It owns its complete
47-file `include/`, `config/`, `tests/`, `v3.mk` and provenance closure. Its single
public version header is
`include/embedding/encoders/causal_patch_bottleneck_mae/versions/v3.h`, relative
to that root. It has no historical model-import API and no implementation
forwarding into another version. Future versions require independent full copies.

V3 starts all learned weights fresh at seed 943001, with fresh TRAIN-only input
and bank statistics; source_updates=0 and causal_updates=2048. The application
uses frozen native36 inference with genuine raw F64, masks, actual timestamps,
semantic IDs and explicit cutoffs. Future values and masks are removed before
any observation processing; temporal attention is causal.

Research training, causal proofs, typed resume and initial/final fixed-head
measurements are complete. Quality is mixed and no overall winner is declared.
The [current integration record](../../../../doc/CAUSAL_PRIMARY_INTEGRATION.md)
separately tracks new application verification. CPB-v1/v2 remain historical
inherited-weight references, never clean-research quality targets. Old dev v1
source/catalog bytes are preserved in the sibling research archive; old records
are not active selectors. All historical checkpoints and results remain protected.
