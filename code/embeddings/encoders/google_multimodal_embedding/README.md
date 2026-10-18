# Google multimodal embedding provider

This external representation source is a documented integration target for
Mityane operations. The selected stable model is **Gemini Embedding 2**,
`gemini-embedding-2`, checked against Google's documentation on 18 October 2026.
Google lists text, image, video, audio and PDF inputs in a shared space, with
flexible output dimensions. See the [model reference](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2)
and [general-availability announcement](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2-generally-available/).

[provider.json](provider.json) records the chosen service identity and proposed
3072-dimensional output contract. Google's [embedding guide](https://ai.google.dev/gemini-api/docs/embeddings)
documents that default output size and request conventions. The public service
identifier is not an immutable checkpoint hash; a future adapter must retain
the returned revision when available, request configuration, execution date
and response artifact identity.

## Current state

The provider is registered here for planning and source provenance. Its API
adapter and operation evaluation remain proposed. No credentials were inspected,
SDK installed, media uploaded, API request executed or weights imported for
this registration. Foundation-model training is outside this workstream.

## Integration work

The adapter should accept source-referenced text or media observations and return
embedding records tied to an explicit `SpaceRef`. Preserve modality, grouping,
source hashes, task instructions, model identity, dimensions, normalization and
request/response provenance. Handle truncation or unsupported content explicitly.
Changes of model, task convention or dimensions create distinct space identities.

Keep credentials in the supported external credential mechanism. Document the
chosen API route, service terms, request budgets, retry limits and cache policy
before live use. Use the [Gemini API service terms](https://ai.google.dev/gemini-api/terms)
as the service reference; the repository's MIT license does not supply a license
for provider weights or authorization to transmit datasets.

Begin operation evidence with a bounded multimodal retrieval fixture and
replayable recorded responses. Test source identity, shape, missing/unsupported
inputs and task conventions before comparing retrieval against an appropriate
baseline. No local operation quality has been measured for this provider.
