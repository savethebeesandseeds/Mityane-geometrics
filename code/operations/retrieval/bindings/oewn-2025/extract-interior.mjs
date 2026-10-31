// Freeze the OR01-RET-ABS 0.2.0 English noun hypernym interior.
// This script does not embed text, call a provider, or rank neighbors.
// It does not write the 0.1.0 snapshot.
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";

const EXPECTED_ARCHIVE_SHA256 = "7d749f6e2c39e6970e4997839dcf6e42fd281f3c2fae0171d2192bae8cfa4b51";
const ROOT_ID = "00001740-n";

function fail(message) {
  process.stderr.write(`oewn interior: ${message}\n`);
  process.exit(1);
}

function argument(name) {
  const index = process.argv.indexOf(name);
  if (index < 0 || index + 1 >= process.argv.length) fail(`missing ${name}`);
  return process.argv[index + 1];
}

function sha256(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

const dumpDir = argument("--dump");
const zipPath = argument("--zip");
const outPath = argument("--out");
const archiveSha = sha256(zipPath);
if (archiveSha !== EXPECTED_ARCHIVE_SHA256) {
  fail(`archive sha256 ${archiveSha} does not match ${EXPECTED_ARCHIVE_SHA256}`);
}

const synsets = new Map();
const lexfiles = new Map();
const children = new Map();
for (const name of readdirSync(dumpDir).sort()) {
  if (!name.startsWith("noun.") || !name.endsWith(".json")) continue;
  const lexfile = basename(name, ".json");
  for (const [id, row] of Object.entries(JSON.parse(readFileSync(join(dumpDir, name), "utf8")))) {
    if (synsets.has(id)) fail(`duplicate synset ${id}`);
    if (row.instance_hypernym && row.instance_hypernym.length > 0) {
      fail(`instance_hypernym is present on ${id}`);
    }
    synsets.set(id, row);
    lexfiles.set(id, lexfile);
    if (!children.has(id)) children.set(id, []);
  }
}

const targeted = new Set();
for (const [id, row] of synsets) {
  const parents = [...(row.hypernym || [])].sort();
  for (const parent of parents) {
    if (!synsets.has(parent)) fail(`missing hypernym ${parent} from ${id}`);
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent).push(id);
    targeted.add(parent);
  }
}

if (!synsets.has(ROOT_ID)) fail(`missing root ${ROOT_ID}`);
const depth = new Map([[ROOT_ID, 0]]);
const queue = [ROOT_ID];
for (let index = 0; index < queue.length; index += 1) {
  const node = queue[index];
  const here = depth.get(node);
  for (const child of children.get(node) || []) {
    if (depth.has(child)) continue;
    depth.set(child, here + 1);
    queue.push(child);
  }
}
if (depth.size !== synsets.size) fail(`hypernym graph reached ${depth.size} of ${synsets.size}`);

const abstractCorpus = [...targeted].sort().map((sourceId) => {
  const row = synsets.get(sourceId);
  const parents = [...(row.hypernym || [])].sort();
  return {
    source_id: sourceId,
    members: row.members || [],
    definition: row.definition || [],
    ili: row.ili || null,
    lexfile: lexfiles.get(sourceId),
    population: "english-abstract",
    depth_from_entity: depth.get(sourceId),
    lineage: parents.map((parentId) => ({ relation: "hypernym", parent_id: parentId })),
  };
});

const snapshot = {
  binding_id: "OR01-RET-ABS",
  revision: "0.2.0",
  profile: "local-saved-vectors",
  language: "en",
  admits_run: false,
  contract: "OR01-RET",
  contract_revision: "0.1.0",
  requirements_record: "OR01-RET-RER",
  requirements_revision: "0.1.0",
  population: "english-noun-hypernym-interior",
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
    rule: "A noun synset is abstract when at least one other noun synset names it as a hypernym. Leaves are outside the corpus.",
    excluded_relations: ["instance_hypernym"],
    part_of_speech: "n",
    root_id: ROOT_ID,
    minimal_population: "OR01-RET-ABS 0.1.0",
  },
  temporal: { mode: "non_temporal", reason: "The ontology snapshot has no observation time." },
  counts: {
    noun_synsets: synsets.size,
    abstract_entries: abstractCorpus.length,
    leaves_excluded: synsets.size - abstractCorpus.length,
  },
  abstract_corpus: abstractCorpus,
};

writeFileSync(outPath, `${JSON.stringify(snapshot, null, 2)}\n`);
process.stdout.write(
  `nouns=${synsets.size} abstract=${abstractCorpus.length} leaves=${synsets.size - abstractCorpus.length}\n`,
);
