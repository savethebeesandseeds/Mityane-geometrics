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

## Agent instructions: reusable provider tools

Follow the shared [encoder Hero instructions](../AGENTS.md). Hx18 develops a
reusable provider adapter, configuration and validation surface that operations
and applications can compose. Keep the provider/model identity, supported
request configuration, evaluation recipe and each recorded response distinct.
A selected model or output width supplies one integration binding; it does not
define a finished application or establish downstream usefulness.

Expose only capabilities and configuration ranges the upstream service actually
supports. Preserve modality, dimensions, space identity and provenance across
the public interfaces, and document implementation gaps and service limits.
This external-provider workstream does not imply access to weights, local
fresh training or arbitrary changes to the provider architecture. Applications
such as Cuwacunu own final docking and application-specific work. Preserve
historical provider/evaluation records; this guidance authorizes no live API
request, upload or new experiment.

## Current state

The provider is registered here for planning and source provenance. Its API
adapter and operation evaluation remain proposed. No credentials were inspected,
SDK installed, media uploaded, API request executed or weights imported for
this registration. Foundation-model training is outside this workstream.

## Local embedding file

Saved embeddings for this hero are kept in IdyDB, a copied flat-file library
at [code/embeddings/idydb](../../idydb/MITYANE.md). The copy arrived on 30
October 2026 from `C:\Work\idydb`. It stores text and float vectors. Its own
cosine and L2 query is a nearest-neighbor helper inside that library. It does
not implement the Gemini adapter, and it does not authorize a live API call.
Vector files stay outside Git.

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
