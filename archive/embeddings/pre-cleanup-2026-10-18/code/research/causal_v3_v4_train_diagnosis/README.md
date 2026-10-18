# Saved TRAIN v3/v4 comparison

Prospective protocol:
doc/CAUSAL_V3_V4_TRAIN_DIAGNOSIS_CARD.json and .md.

This folder owns physical copies of the prior neutral scalar helpers and inert
C++ tensor-data reader. It imports no old source, Torch or NumPy and runs no
learned module. Explicit41 input IDs and per-file field allowlists forbid held-out
tensor access or neural weights. Aggregate JSON is parsed but only TRAIN is used.

After root source/card approval and preservation, inside existing cuwacunu only:

    python3 schema.py /ABS/SOURCE/CAUSAL_V3_V4_TRAIN_DIAGNOSIS_CARD.json
    python3 diagnose.py --card /ABS/SOURCE/CAUSAL_V3_V4_TRAIN_DIAGNOSIS_CARD.json \
      --output /ABS/NEW_DIAGNOSIS_LEAF

schema.py reads metadata only. diagnose.py writes an exclusive diagnosis.json or
preserved failure.json. No later encoder/objective is declared or selected here.
