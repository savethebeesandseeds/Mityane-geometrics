// Freeze the OR01-RET-ABS 0.1.0 English text population from Open English WordNet 2025.
// This script does not embed text, call a provider, or rank neighbors.
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";

const EXPECTED_ARCHIVE_SHA256 = "7d749f6e2c39e6970e4997839dcf6e42fd281f3c2fae0171d2192bae8cfa4b51";
const DEPTH_MAX = 3;

function fail(message) {
  process.stderr.write(`oewn extract: ${message}\n`);
  process.exit(1);
}

function argument(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) fail(`missing ${name}`);
  return process.argv[index + 1];
}

function loadJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function combinations(items, size) {
  const found = [];
  function walk(start, chosen) {
    if (chosen.length === size) {
      found.push(chosen.slice());
      return;
    }
    for (let index = start; index < items.length; index += 1) {
      chosen.push(items[index]);
      walk(index + 1, chosen);
      chosen.pop();
    }
  }
  walk(0, []);
  return found;
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

const dumpDir = argument("--dump");
const zipPath = argument("--zip");
const seedsPath = argument("--seeds");
const outPath = argument("--out");

const archiveSha = sha256(zipPath);
if (archiveSha !== EXPECTED_ARCHIVE_SHA256) {
  fail(`archive sha256 ${archiveSha} does not match ${EXPECTED_ARCHIVE_SHA256}`);
}

const seedFile = loadJson(seedsPath);
if (seedFile.binding_id !== "OR01-RET-ABS" || seedFile.revision !== "0.1.0") {
  fail("seeds.json is not OR01-RET-ABS 0.1.0");
}
if (seedFile.archive_sha256 !== EXPECTED_ARCHIVE_SHA256) fail("seeds.json archive hash disagrees");
if (!Array.isArray(seedFile.seeds) || seedFile.seeds.length === 0) fail("seed list is empty");

const synsets = new Map();
const lexfiles = new Map();
const entries = new Map();
for (const name of readdirSync(dumpDir).sort()) {
  if (name.startsWith("noun.") && name.endsWith(".json")) {
    const lexfile = basename(name, ".json");
    for (const [id, row] of Object.entries(loadJson(join(dumpDir, name)))) {
      if (synsets.has(id)) fail(`duplicate synset ${id}`);
      synsets.set(id, row);
      lexfiles.set(id, lexfile);
    }
  } else if (name.startsWith("entries-") && name.endsWith(".json")) {
    for (const [lemma, row] of Object.entries(loadJson(join(dumpDir, name)))) {
      if (entries.has(lemma)) fail(`duplicate entry ${lemma}`);
      entries.set(lemma, row);
    }
  }
}

const seenIds = new Set();
const concrete = [];
for (const seed of seedFile.seeds) {
  if (seenIds.has(seed.synset_id)) fail(`duplicate seed ${seed.synset_id}`);
  seenIds.add(seed.synset_id);
  const entry = entries.get(seed.lemma);
  const senses = entry && entry.n && Array.isArray(entry.n.sense) ? entry.n.sense : null;
  if (!senses) fail(`no noun entry for ${seed.lemma}`);
  const sense = senses.find((row) => row.id === seed.sense_key);
  if (!sense) fail(`sense ${seed.sense_key} is absent`);
  if (sense.synset !== seed.synset_id) {
    fail(`sense ${seed.sense_key} points at ${sense.synset}, not ${seed.synset_id}`);
  }
  const row = synsets.get(seed.synset_id);
  if (!row) fail(`missing seed synset ${seed.synset_id}`);
  concrete.push({
    source_id: seed.synset_id,
    lemma: seed.lemma,
    sense_key: seed.sense_key,
    members: row.members || [],
    definition: row.definition || [],
    ili: row.ili || null,
    lexfile: lexfiles.get(seed.synset_id),
    population: "concrete-seed",
  });
}

function pathsFrom(seedId) {
  const found = [];
  function walk(path) {
    const node = path[path.length - 1];
    const row = synsets.get(node);
    if (!row) fail(`missing synset ${node}`);
    const next = [...(row.hypernym || [])].sort();
    for (const hypernym of next) {
      if (path.includes(hypernym)) fail(`hypernym cycle at ${hypernym}`);
      if (!synsets.has(hypernym)) fail(`missing hypernym ${hypernym} from ${node}`);
      const nextPath = path.concat(hypernym);
      const depth = nextPath.length - 1;
      if (depth > DEPTH_MAX) continue;
      found.push(nextPath);
      if (depth < DEPTH_MAX) walk(nextPath);
    }
  }
  walk([seedId]);
  return found;
}

const closures = new Map();
const lineage = new Map();
for (const seed of concrete) {
  const closure = new Set();
  for (const path of pathsFrom(seed.source_id)) {
    for (let index = 1; index < path.length; index += 1) {
      const sourceId = path[index];
      closure.add(sourceId);
      if (!lineage.has(sourceId)) lineage.set(sourceId, []);
      lineage.get(sourceId).push({
        seed_id: seed.source_id,
        depth: index,
        path: path.slice(0, index + 1),
      });
    }
  }
  closures.set(seed.source_id, closure);
}

const abstractCorpus = [...lineage.keys()].sort().map((sourceId) => {
  const row = synsets.get(sourceId);
  const parents = lineage.get(sourceId).slice().sort((left, right) => {
    const seedOrder = left.seed_id.localeCompare(right.seed_id);
    if (seedOrder !== 0) return seedOrder;
    return left.path.join(">").localeCompare(right.path.join(">"));
  });
  return {
    source_id: sourceId,
    members: row.members || [],
    definition: row.definition || [],
    ili: row.ili || null,
    lexfile: lexfiles.get(sourceId),
    population: "english-abstract",
    lineage: parents,
  };
});

const seedIds = concrete.map((seed) => seed.source_id).sort();
const queries = [];
for (const size of [2, 3]) {
  for (const combo of combinations(seedIds, size)) {
    let relevant = null;
    for (const sourceId of combo) {
      const closure = closures.get(sourceId);
      relevant = relevant === null ? new Set(closure) : new Set([...relevant].filter((id) => closure.has(id)));
    }
    const relevantIds = [...relevant].sort();
    queries.push({
      query_id: combo.join("+"),
      concrete_source_ids: combo,
      eligible_for_task_association: relevantIds.length > 0,
      relevant_source_ids: relevantIds,
    });
  }
}

const eligible = queries.filter((query) => query.eligible_for_task_association).length;
const snapshot = {
  binding_id: "OR01-RET-ABS",
  revision: "0.1.0",
  profile: "local-saved-vectors",
  language: "en",
  admits_run: false,
  contract: "OR01-RET",
  contract_revision: "0.1.0",
  requirements_record: "OR01-RET-RER",
  requirements_revision: "0.1.0",
  space_ref: {
    status: "unbound",
    encoder_or_generator: null,
    revision: null,
    dimensions: null,
    layout: null,
    normalization: null,
    metric: null,
    task_or_role_pair: "english-abstract-retrieval",
    saved_vectors: null,
  },
  source: {
    name: "Open English WordNet",
    edition: "2025",
    release_date: "2025-12-31",
    archive: "english-wordnet-2025-json.zip",
    archive_sha256: archiveSha,
    url: "https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025-json.zip",
    license: "CC-BY-4.0",
    citation: "John P. McCrae, Alexandre Rademaker, Francis Bond, Ewa Rudnicka, and Christiane Fellbaum (2019). English WordNet 2019 – An Open-Source WordNet for English.",
  },
  extraction: {
    relation: "hypernym",
    depth_min: 1,
    depth_max: DEPTH_MAX,
    excluded_relations: ["instance_hypernym"],
    part_of_speech: "n",
    concrete_seeds_excluded_from_corpus: true,
  },
  temporal: { mode: "non_temporal", reason: "The ontology snapshot has no observation time." },
  exclusions: {
    duplicate_source_id: "refused",
    self_match: "not_applicable_across_populations",
    concrete_seeds_in_abstract_corpus: "excluded",
  },
  relevance_rule: {
    id: "oewn-shared-hypernym-depth-1-3",
    kind: "synthetic_shared_principle",
    statement: "A concrete query set is relevant to an abstract synset exactly when that synset lies in the intersection of the seeds' hypernym closures at depth 1 through 3. An empty intersection is ineligible for task association.",
    not_relevance: ["distance", "similarity", "cone_membership", "cone_purity"],
  },
  baselines_declared_not_run: [
    {
      id: "plain-exact-neighbor-of-first-seed",
      description: "Exact k-retrieval using the saved vector of the first concrete source id, in ascending id order, as the query point on this abstract corpus.",
    },
    {
      id: "lexical-token-overlap",
      description: "Rank abstract entries by the count of shared lowercase letter-tokens between the joined seed lemmas and the entry members plus definitions, then by ascending source id. This baseline reads text only.",
    },
  ],
  counts: {
    concrete_seeds: concrete.length,
    abstract_entries: abstractCorpus.length,
    queries: queries.length,
    task_association_eligible_queries: eligible,
  },
  concrete_catalog: concrete.slice().sort((left, right) => left.source_id.localeCompare(right.source_id)),
  abstract_corpus: abstractCorpus,
  queries,
};

writeFileSync(outPath, `${JSON.stringify(snapshot, null, 2)}\n`);
process.stdout.write(
  `abstract=${abstractCorpus.length} queries=${queries.length} eligible=${eligible}\n`,
);
