# NSW archival reference data, version 1

This adapter owns data import and mechanical binding for the **wooldridge
`jtrain2` dataset, package 1.4-7**, source commit
`427d117f3888cd3a630ee22a1aedc8f036d128ff`. It supplies an actual non-synthetic
source snapshot for separately authored Hx06 profiles. It does not run matching,
estimate an effect, train an encoder, fit a readout or compute outcome summaries.

The [source lock](source-lock.json) fixes the maintainer, revision, version,
retrieval URLs, source schema and complete population. Original downloaded bytes,
their HTTP locators, retrieval times, byte counts and SHA256 identities are saved
in an exclusive ignored import leaf. No raw source or generated row data belong
in Git.

## Primary provenance and terms

The maintainer's commit-pinned
[DESCRIPTION](https://raw.githubusercontent.com/JustinMShea/wooldridge/427d117f3888cd3a630ee22a1aedc8f036d128ff/DESCRIPTION)
declares **GPL-3**. The
[dataset documentation](https://raw.githubusercontent.com/JustinMShea/wooldridge/427d117f3888cd3a630ee22a1aedc8f036d128ff/R/jtrain2.R)
attributes the data to R. J. LaLonde (1986), *Evaluating the Econometric
Evaluations of Training Programs with Experimental Data*, and describes random
assignment in the National Supported Work Demonstration. Justin M. Shea
maintains the R package. The package's licensing declaration is retained for
these imported data and documentation; this repository's MIT source license
does not replace it. The importer and materializer themselves are authored
project code under the repository license.

The [original Dehejia/Wahba source page](https://users.nber.org/~rdehejia/data/.nswdata3.html)
documents the 1974-earnings subset selected using pre-intervention variables,
with 185 treated and 260 control observations. It distinguishes this experimental
subset from the separate nonexperimental CPS/PSID comparison groups. Those
comparison files are not imported here. The public source page is provenance,
not an original randomization log or a certificate for all subsequent analyses.

The import retains commit-pinned `DESCRIPTION`, `R/jtrain2.R` and
`data/jtrain2.RData`, the maintainer's R-universe package metadata and CSV
serialization, the original author page and the GPL version 3 text. R-universe's
metadata must agree with the fixed package version, source commit, license and
dataset schema. Its CSV is retained as an independently identified serialization.
**The importer does not decode RData or independently establish numeric
equivalence between that binary source and the service CSV.**

This is a widely published retrospective archival benchmark. It is not fresh,
unexposed or protected assessment data. Historical period labels do not establish
original decision-time information availability, exact assignment probabilities,
or a prospective protocol. Source attribution, GPL terms and dataset provenance
must accompany subsequent permitted reuse. Local retrieval publishes neither
the data nor generated row artifacts.

## Source and role definitions

All 19 source fields, in source order, are retained without conversion:

`train`, `age`, `educ`, `black`, `hisp`, `married`, `nodegree`, `mosinex`, `re74`,
`re75`, `re78`, `unem74`, `unem75`, `unem78`, `lre74`, `lre75`, `lre78`, `agesq`,
`mostrn`.

| Bound role | Source definition and limit |
| --- | --- |
| Unit | One source CSV row; exactly 445 rows. No persistent subject identifier is supplied by this adapter. |
| Assignment | `train`: 1 assigned to job training, 0 control; complete source roster is 185/260. No assignment probability is invented. |
| Allowed matching descriptor | `re74`: observed real earnings during 1974, in thousands of dollars, as documented by the package. Only this finite nonnegative coordinate is admitted. |
| Outcome definition | `re78`: observed real earnings during 1978, in thousands of dollars. Finite nonnegative source values are required; outcome values remain exclusively in the original CSV. No mean, contrast or other outcome aggregate is computed. |
| Other columns | Retained original evidence only. Retention does not admit them as matching, adjustment, representation or selection inputs. |
| Availability | Unknown (`available_at: null`). Retrospective baseline role does not assert availability at an original decision cutoff. |

The [baseline representation definition](binding-definition.json) declares
provider `nsw-re74-identity/v1`, a dense native `[1]` binary64 vector containing
exactly the parsed `re74` value, without normalization, masks, fitting, adapters,
selection or readout. Its geometry is `absolute_distance/v1`, measured in the
same units. This transparent raw descriptor makes no learned-encoder utility
claim.

Materialization writes actual provider, SpaceRef, assignment roster and descriptor
artifacts with digest links to the CSV and each other. Source row ordinals supply
stable lineage within this exact snapshot namespace; they are not matching
features or cross-dataset person identities. Derived artifacts contain no
`re78` values. An operation instance must still bind its estimand or descriptive
question, permitted matching rule, support, assumptions, refusal policy and
evaluation. The materials here establish no effect or scientific quality result.

## Execution and preservation

Use the existing managed container and pinned Node runtime, without installing
R, Python packages or another toolchain. A local data-only admission record at
`.temp/data/hx06-nsw-reference-v1/data-admission-001.json` must exist before
retrieval. It binds the user's authorized data-source/baseline preparation scope
and explicitly excludes matching, fitting, effect estimation and outcome
aggregation.

```powershell
.\container.ps1 -Action exec -Command @('bash','code/scripts/node.sh','--test','code/data/observations/open_benchmarks/nsw_wooldridge_v1/import.test.mjs')
.\container.ps1 -Action exec -Command @('bash','code/scripts/node.sh','code/data/observations/open_benchmarks/nsw_wooldridge_v1/import.mjs','--output','.temp/data/hx06-nsw-reference-v1/import-001')
.\container.ps1 -Action exec -Command @('bash','code/scripts/node.sh','code/data/observations/open_benchmarks/nsw_wooldridge_v1/materialize.mjs','--import','.temp/data/hx06-nsw-reference-v1/import-001','--output','.temp/data/hx06-nsw-reference-v1/binding-preparation-001')
```

Those two leaves were successfully created on **20 October 2026**. The commands
are documentation of that execution, not instructions to overwrite or replay
it. Existing or partial leaves are preserved and refused. Any permitted future
retrieval needs a new numbered leaf; verify its observed artifact hashes against
the retained receipt before claiming the same snapshot. An updated service
serialization or publisher revision creates a new artifact identity even when
the package label is unchanged.

The exact first CSV is 37,008 bytes, SHA256
`3dab3d903e07549343ae3aa51ce6530afc490c5861f6ebec76360e48fed9f9df`.
The import receipt is 8,909 bytes, SHA256
`ca3c84929719643c43e17d2f0e0c98d259bf12538539cfedfdec11be58794c1a`.
The materialization receipt links actual provider, SpaceRef, roster and
representation hashes. These records and all seven original downloads remain
under the ignored `.temp/data/hx06-nsw-reference-v1/` tree.

Six artificial tests cover malformed CSV and schema, row/assignment changes,
nonfinite or missing role fields, source metadata mismatches, prohibited
covariate additions, and isolation of the baseline descriptor from all other
retained columns. Passing these import tests is an engineering result only.
