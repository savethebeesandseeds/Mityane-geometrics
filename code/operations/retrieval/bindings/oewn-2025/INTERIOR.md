# OR01-RET-ABS 0.2.0

English noun hypernym interior for Hx01 geometric combination. This record freezes the text population and the parent lineage. It does not replace [OR01-RET-ABS 0.1.0](BINDING.md). That revision remains the minimal ten-seed, depth-1-through-3 population of 29 abstract entries. This revision does not embed text, rank neighbors, or admit a relevance result.

| Field | Value |
| --- | --- |
| Binding | `OR01-RET-ABS`, revision `0.2.0` |
| Operation | [OR01-RET 0.1.0](../../CONTRACT.md) |
| Population | `english-noun-hypernym-interior` |
| Language | English |
| Claim admitted by this record | none |
| Snapshot | [interior-snapshot.json](interior-snapshot.json), sha256 `7968b72a5276ce3572f6df070fb45aa74f8186e25283edea03279b07db17433a` |

## Extraction

The source archive is the same Open English WordNet 2025 zip named in the 0.1.0 binding, sha256 `7d749f6e2c39e6970e4997839dcf6e42fd281f3c2fae0171d2192bae8cfa4b51`. The project does not vendor that archive.

The relation is noun `hypernym` only. `instance_hypernym` is excluded. The 2025 noun files contained no `instance_hypernym` field. A noun synset is an abstract entry when at least one other noun synset names it as a hypernym. A leaf is outside the corpus. The graph is rooted at `00001740-n` (`entity`). Every noun synset is reachable from that root.

| Count | Value |
| --- | --- |
| Noun synsets | 71,864 |
| Abstract entries | 17,191 |
| Leaves excluded | 54,673 |

Every abstract entry keeps its synset id, member lemmas, definition, ILI, lexfile, depth from `entity`, and its direct hypernym parents. Texts are the WordNet wording. The ten 0.1.0 seeds sit inside this interior because each is itself a hypernym of other nouns. They are corpus members here, not the query population. A geometric query for this population is a noun leaf, which is outside the corpus.

`admits_run` is false. Saved vectors for these 17,191 texts are not part of this revision. The 0.1.0 snapshot sha256 is unchanged: `ce66b539ec47ab3380e076230c3fc55db4520b1c54f3f86d69e2c6223f16338c`.

## Reproduction

```powershell
.\container.ps1 exec bash code/scripts/node.sh code/operations/retrieval/bindings/oewn-2025/extract-interior.mjs --dump /tmp/oewn-2025-json --zip /tmp/english-wordnet-2025-json.zip --out code/operations/retrieval/bindings/oewn-2025/interior-snapshot.json
```

A regeneration must match the snapshot sha256 above. Open English WordNet is released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite John P. McCrae, Alexandre Rademaker, Francis Bond, Ewa Rudnicka, and Christiane Fellbaum (2019), *English WordNet 2019 – An Open-Source WordNet for English*.
