# OR01-RET-ABS 0.1.0

English abstract corpus binding for Hx01 geometric retrieval. This record freezes the text population, the parent lineage, the shared-principle labels, and one local text SpaceRef with its saved-vector hashes. It selects the `local-saved-vectors` profile. The score of that population is a separate record, [OR01-RET-ABS-EVAL 0.1.0](PROTOCOL.md).

| Field | Value |
| --- | --- |
| Binding | `OR01-RET-ABS`, revision `0.1.0` |
| Operation | [OR01-RET 0.1.0](../../CONTRACT.md) |
| Requirements | [OR01-RET-RER 0.1.0](../../EVIDENCE.md) |
| Profile | `local-saved-vectors` |
| Language | English |
| Claim admitted by this record | none |
| Later claim levels this population can use | `exploratory_description` or `task_association` |
| Snapshot | [snapshot.json](snapshot.json), sha256 `ce66b539ec47ab3380e076230c3fc55db4520b1c54f3f86d69e2c6223f16338c` |

## 1. Ontology and extraction

The source is the Open English WordNet 2025 edition, released 31 December 2025, archive `english-wordnet-2025-json.zip`, sha256 `7d749f6e2c39e6970e4997839dcf6e42fd281f3c2fae0171d2192bae8cfa4b51`. The project does not vendor that archive. The frozen population is the extracted snapshot only.

The extraction follows the noun `hypernym` relation and no other relation. `instance_hypernym` is excluded. From each pinned seed synset, every hypernym path of depth 1, 2, or 3 is kept. Depth 0, the seed itself, is not an abstract entry. A missing hypernym target or a cycle fails the extraction. Hypernym identifiers are expanded in ascending synset-id order. The same abstract synset reached from several seeds is one corpus member, and its lineage lists every path.

The seed is a pinned synset, not whichever sense is listed first. These ten noun synsets are the concrete catalog:

| Lemma | Sense key | Synset |
| --- | --- | --- |
| dog | `dog%1:05:00::` | `02086723-n` |
| wolf | `wolf%1:05:00::` | `02116752-n` |
| cat | `cat%1:05:00::` | `02124272-n` |
| bicycle | `bicycle%1:06:00::` | `02837983-n` |
| car | `car%1:06:00::` | `02961779-n` |
| chisel | `chisel%1:06:00::` | `03024375-n` |
| hammer | `hammer%1:06:00::` | `03486255-n` |
| knife | `knife%1:06:00::` | `03628765-n` |
| bread | `bread%1:13:00::` | `07695101-n` |
| apple | `apple%1:13:00::` | `07755101-n` |

`hammer%1:06:00::` is the hand tool. The gunlock sense of hammer and the proverb sense of saw are not seeds. The declared list is [seeds.json](seeds.json). [extract.mjs](extract.mjs) checks each sense key against the archive and writes the snapshot. It does not embed text or rank neighbors.

The snapshot holds 10 concrete seeds and 29 abstract entries. The two sets are disjoint. Every abstract entry keeps its synset id, member lemmas, definition, ILI, lexfile, and the seed paths that reached it. Texts are the WordNet wording, copied from that archive.

## 2. Relevance rule

The rule id is `oewn-shared-hypernym-depth-1-3`. It is a synthetic shared-principle label taken from the ontology graph, and it was fixed before any vector existed.

A query in this revision is a set of two or three concrete seeds. The set is relevant to an abstract synset exactly when that synset lies in the intersection of those seeds' depth-1-through-3 hypernym closures. An empty intersection is ineligible for `task_association`. Distance, similarity, cone membership, and cone purity are not this rule.

There are 165 such sets. Seven are eligible:

| Query | Concrete seeds | Relevant abstract entries |
| --- | --- | --- |
| `02086723-n+02116752-n` | dog, wolf | placental, carnivore, canine |
| `02086723-n+02124272-n` | dog, cat | placental, carnivore |
| `02116752-n+02124272-n` | wolf, cat | placental, carnivore |
| `02837983-n+02961779-n` | bicycle, car | wheeled vehicle |
| `03024375-n+03628765-n` | chisel, knife | cutter, cutting implement, edge tool |
| `07695101-n+07755101-n` | bread, apple | solid food |
| `02086723-n+02116752-n+02124272-n` | dog, wolf, cat | placental, carnivore |

The other 158 sets are listed in the snapshot with an empty relevant set. Hammer has hypernyms inside the abstract corpus and has no eligible partner among these seeds at depth 3 or less. Those hypernyms remain corpus members. They are not a task-association label for any query in this revision. A single concrete point is not a task-association query here. A later exploratory description of one point would still have to be labeled exploratory.

## 3. SpaceRef

One SpaceRef covers the concrete catalog and the abstract corpus. The bytes are a local artifact. Git holds the hashes.

| Field | Value |
| --- | --- |
| Status | frozen for this population |
| Profile | `local-saved-vectors` |
| Encoder | `google/embeddinggemma-2`, text only |
| Revision | `f9972ac22e696a105c482202646979e33e4b5a1e` |
| Loader | Transformers text tower. `vision_config` and `audio_config` are unset. The checkpoint's sentence-transformers entry imports a video processor, so the vectors were not produced through that entry |
| Runtime | torch `2.6.0+cu124`, transformers `5.22.1`, CUDA device NVIDIA RTX A2000 8GB Laptop GPU `GPU-a878a4ab-d3a8-a4d5-8e6f-010b9dd24596`, dtype `bfloat16` |
| Dimensions | 768, native, no truncation |
| Layout | one vector per source id |
| Normalization | `l2` |
| Metric | `cosine` |
| Task or role pair | prompt `SentenceSimilarity`, text `task: sentence similarity | query: ` |
| Surface | `{member}: {definition}`, first member and first definition, then that prompt |
| Space id | `local-text-768-cosine-l2-sentence-similarity` |
| Snapshot id | `or01-ret-abs-0.1.0-local-text` |
| Pooling | mean over tokens, prompt included, then L2 normalization |
| Population role | `english-abstract-retrieval` |
| Concrete vectors | `.temp/operations/retrieval/or01-ret-abs-0.1.0-local-text/concrete.vec`, sha256 `6c8e7bccf3b2e16cdb01abe1a80489d266b167954dfe035271ee30e8fe90e885`, 10 rows |
| Abstract vectors | `.temp/operations/retrieval/or01-ret-abs-0.1.0-local-text/abstract.vec`, sha256 `0096a2d4338d03ba94973d2679ad0604a0003958985208171acc13f2da535a81`, 29 rows |
| Text snapshot | [snapshot.json](snapshot.json), sha256 `ce66b539ec47ab3380e076230c3fc55db4520b1c54f3f86d69e2c6223f16338c` |

The model card states Apache 2.0. This SpaceRef is EmbeddingGemma 2 text. It is not Gemini Embedding 2 and it does not close Hx18. The encoder was not trained. No live API call was made. Equal length is not the binding: the fields above are.

## 4. Exclusions, time, and baselines

Duplicate source ids are refused. The concrete seeds are not members of the abstract corpus, so self-match does not apply across the two populations. The temporal mode is `non_temporal` because this ontology snapshot has no observation time.

Two baselines are named here. [PROTOCOL.md](PROTOCOL.md) fixes how they are scored. [oewn-abstract-evaluation-v0.1.0.md](../../reviews/oewn-abstract-evaluation-v0.1.0.md) records the comparison.

- Plain exact nearest neighbor, using the saved vector of the first concrete source id in ascending order as the query point on this abstract corpus.
- Lexical token overlap: shared lowercase ASCII letter-tokens between the pinned seed lemmas and an abstract entry's members plus definitions, with remaining ties broken by ascending source id. That baseline reads text only.

## 5. Limit

This binding links OR01-RET 0.1.0 and OR01-RET-RER 0.1.0. It does not revise either record. It does not admit a relevance result or a Gemini call. The encoder forward pass that wrote the saved vectors is the SpaceRef above. It is not training and it is not a relevance run. An exploratory neighborhood listing is [oewn-abstract-neighborhoods-v0.1.0.md](../../reviews/oewn-abstract-neighborhoods-v0.1.0.md). A result on this population would not transfer to another encoder, to a Nonverba evaluation, or to a contrast in Hx06. Hubness and cone purity remain diagnostics and do not close Hx02. English is the only language. A multilingual expansion waits on explicit alignment diagnostics.

Open English WordNet is derived from Princeton WordNet and released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite John P. McCrae, Alexandre Rademaker, Francis Bond, Ewa Rudnicka, and Christiane Fellbaum (2019), *English WordNet 2019 – An Open-Source WordNet for English*.

## 6. Reproduction

Regeneration needs the named archive, checked against the sha256 above, and the managed container:

```powershell
.\container.ps1 exec bash code/scripts/node.sh code/operations/retrieval/bindings/oewn-2025/extract.mjs --dump /tmp/oewn-2025-json --zip /tmp/english-wordnet-2025-json.zip --seeds code/operations/retrieval/bindings/oewn-2025/seeds.json --out code/operations/retrieval/bindings/oewn-2025/snapshot.json
```

The dump directory is the unzipped archive and is not part of the repository. The committed snapshot is the frozen text. A regeneration must match its sha256.

The saved vectors were written in the managed container by [encode_saved_vectors.py](encode_saved_vectors.py), with `HF_HOME` and the Python prefix under `/opt/mityane-geometrics/producers/embeddinggemma-text`. Model weights stay on that container disk. The vector files stay under ignored `.temp/`. A regeneration must match the concrete and abstract sha256 values in the SpaceRef.
