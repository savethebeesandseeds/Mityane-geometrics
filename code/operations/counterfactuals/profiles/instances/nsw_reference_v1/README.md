# NSW archival profile bindings

**Binding set:** `HX06-NSW-REFERENCE`, revision `1.0.0`, 20 October 2026.

This binds two separately identified non-synthetic instances to the newly
authored profiles. It does not implement either operation or report a new
comparison/effect. The [observational instance](OBSERVATIONAL.md) and
[intervention instance](INTERVENTION.md) have separate RER IDs, targets,
assumptions and refusal policies. Their exact artifacts and permissions are in
[bindings.json](bindings.json).

## Source and rights

The source is `wooldridge::jtrain2` version `1.4-7`, maintainer source revision
`427d117f3888cd3a630ee22a1aedc8f036d128ff`. The
[maintainer documentation](https://justinmshea.github.io/wooldridge/reference/jtrain2.html)
describes 445 observations, assignment to job training, 1974/1978 earnings in
thousands of real dollars and randomized training status. The
[original authors' source page](https://users.nber.org/~rdehejia/data/.nswdata3.html)
describes the Dehejia-Wahba subset as selected using pre-intervention variables,
with 185 assigned and 260 control observations.

The [maintainer's package metadata](https://github.com/JustinMShea/wooldridge/blob/427d117f3888cd3a630ee22a1aedc8f036d128ff/DESCRIPTION)
declares GPL-3. Retained source documentation and full license text accompany
the local source bytes. Preserve attribution and applicable license duties;
this binding authorizes local archival preparation/inspection, not publication
of individual records, linkage/re-identification or transmission to an embedding
API. Raw/derived data stay under ignored `.temp/` paths. No commercial or
production use is selected by these instances.

The owning [data importer](../../../../../data/observations/open_benchmarks/nsw_wooldridge_v1/import.mjs)
and [source lock](../../../../../data/observations/open_benchmarks/nsw_wooldridge_v1/source-lock.json)
retain exact publisher URLs, source revision and actual-byte receipt. CSV is the
maintainer's R-universe serialization; the retained RData file is a separate
artifact. No R installation/decoding or independent equivalence assertion is
implied. The actual served CSV hash identifies the analysis input.

## Representation and exposure boundary

The selected representation is the deterministic raw descriptor `[re74]` in
thousands of real dollars, shape `[1]`, dense, with absolute scalar distance.
It uses no learned encoder, preprocessing fit, checkpoint, nuisance model,
readout or selection population. Actual provider/SpaceRef/roster/representation
bytes are hash-bound in the binding record. This is a simple descriptor control,
not evidence of learned geometry quality. The TRAIN-only synthetic OLS binding
is not used, so its keep/revise/replace decision is not a dependency here.

`train`, outcomes, later earnings, participation duration and row identifiers
are not representation coordinates. Snapshot/row identifiers provide provenance
and deterministic tie handling only. `re74` is designated pre-intervention by
the source documentation; individual assignment/measurement availability dates
are not supplied. Mode is retrospective; no historical live/as-of eligibility
or per-person timestamp is invented.

The benchmark and published outcomes were already publicly exposed. These are
archival reference/exploratory bindings, not fresh protected assessment data.
The protocols below freeze any future computation before this project performs
it; they cannot erase public exposure. Import/schema checks and baseline-only
materialization are mechanical preparation, not matched analysis, causal
estimation, training or a scientific verdict.

## Execution boundary

Both instances are bound but scientifically **not admitted**: profile estimators
and their conformance/evaluation do not yet exist. A dependent execution must
retain an actual implementation/source identity, admission and attempt record,
verify source/design requirements, and execute the appropriate protocol before
any scientific claim. Unknown design information triggers the profile's typed
refusal when needed by the requested estimand or uncertainty method. The existing
OR06-PCR CLI continues to reject both new claim levels.
