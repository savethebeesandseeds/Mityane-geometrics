// Score OR01-RET-ABS-EVAL 0.1.0 from a saved-vector inspector listing.
// This script does not embed text, call a provider, or rank geometric queries.
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

const ABSTRACT_SHA256 = "0096a2d4338d03ba94973d2679ad0604a0003958985208171acc13f2da535a81";
const CONCRETE_SHA256 = "6c8e7bccf3b2e16cdb01abe1a80489d266b167954dfe035271ee30e8fe90e885";
const SNAPSHOT_SHA256 = "ce66b539ec47ab3380e076230c3fc55db4520b1c54f3f86d69e2c6223f16338c";
const APERTURES = ["0", "pi/6", "pi/3", "pi/2", "pi"];
const CUTOFFS = [1, 3, 5];

function fail(message) {
  process.stderr.write(`oewn score: ${message}\n`);
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

function tokensOf(text) {
  return text.toLowerCase().match(/[a-z]+/g) ?? [];
}

function tokenSet(texts) {
  const found = new Set();
  for (const text of texts) {
    for (const token of tokensOf(text)) found.add(token);
  }
  return found;
}

function discount(rank) {
  return 1 / Math.log2(rank + 1);
}

function ndcg(order, relevant) {
  const wanted = new Set(relevant);
  let gain = 0;
  order.forEach((id, index) => {
    if (wanted.has(id)) gain += discount(index + 1);
  });
  let ideal = 0;
  for (let rank = 1; rank <= relevant.length; rank += 1) ideal += discount(rank);
  if (ideal === 0) fail("nDCG ideal is zero");
  return gain / ideal;
}

function hitsAt(order, relevant, k) {
  const wanted = new Set(relevant);
  let hits = 0;
  for (let index = 0; index < k; index += 1) {
    if (wanted.has(order[index])) hits += 1;
  }
  return hits;
}

function checkFixture() {
  const perfect = ndcg(["a", "b", "c"], ["a", "b"]);
  if (perfect !== 1) fail(`perfect nDCG is ${perfect}`);
  const late = ndcg(["c", "b", "a"], ["a"]);
  if (!(late < 1)) fail("a relevant item at the end outscored the top");
}

function loadPopulation() {
  const snapshotPath = argument("--snapshot");
  const seedsPath = argument("--seeds");
  const abstractPath = argument("--abstract");
  const concretePath = argument("--concrete");
  if (sha256(snapshotPath) !== SNAPSHOT_SHA256) fail("snapshot hash mismatch");
  if (sha256(abstractPath) !== ABSTRACT_SHA256) fail("abstract vector hash mismatch");
  if (sha256(concretePath) !== CONCRETE_SHA256) fail("concrete vector hash mismatch");
  const snapshot = JSON.parse(readFileSync(snapshotPath, "utf8"));
  const seeds = JSON.parse(readFileSync(seedsPath, "utf8"));
  const lemmas = new Map(seeds.seeds.map((seed) => [seed.synset_id, seed.lemma]));
  const queries = snapshot.queries.filter((query) => query.eligible_for_task_association);
  if (queries.length !== 7) fail(`expected 7 eligible queries, found ${queries.length}`);
  for (const query of queries) {
    if (query.relevant_source_ids.length === 0) fail(`eligible query ${query.query_id} has no labels`);
    for (const id of query.concrete_source_ids) {
      if (!lemmas.has(id)) fail(`missing lemma ${id}`);
    }
  }
  return { snapshot, lemmas, queries };
}

function lexicalOrder(query, lemmas, abstract) {
  const seedTokens = tokenSet(query.concrete_source_ids.map((id) => lemmas.get(id)));
  if (seedTokens.size === 0) fail(`empty seed tokens for ${query.query_id}`);
  const rows = abstract.map((entry) => {
    const overlap = [...tokenSet([...entry.members, ...entry.definition])].filter((token) => seedTokens.has(token)).length;
    return { id: entry.source_id, overlap };
  });
  rows.sort((left, right) => right.overlap - left.overlap || (left.id < right.id ? -1 : left.id > right.id ? 1 : 0));
  return rows.map((row) => row.id);
}

function writeQueries(queries, path) {
  const lines = [];
  for (const query of queries) {
    const ids = query.concrete_source_ids;
    const labels = query.relevant_source_ids.join(" ");
    lines.push(`point ${ids[0]} 29 label ${labels}`);
    if (ids.length === 2) lines.push(`difference ${ids.join(" ")} 29 label ${labels}`);
    lines.push(`shared ${ids.join(" ")} 29 label ${labels}`);
    for (const theta of APERTURES) lines.push(`cone ${ids.join(" ")} theta ${theta} 29 label ${labels}`);
  }
  writeFileSync(path, `${lines.join("\n")}\n`);
}

function expectedBlocks(queries) {
  const blocks = [];
  for (const query of queries) {
    const ids = query.concrete_source_ids;
    blocks.push({ query, system: "point", header: ["point", ids[0]] });
    if (ids.length === 2) blocks.push({ query, system: "difference", header: ["difference", ...ids] });
    blocks.push({ query, system: "shared", header: ["shared", ...ids] });
    for (const theta of APERTURES) {
      blocks.push({ query, system: "cone", header: ["cone", ...ids, "theta", theta], theta });
    }
  }
  return blocks;
}

function parseInspect(text) {
  const blocks = [];
  let current = null;
  function push() {
    if (current) blocks.push(current);
  }
  for (const line of text.split("\n")) {
    if (line === "") continue;
    if (line.startsWith("query ")) {
      push();
      current = { header: line.slice(6).split(" "), hits: [], outcome: null, constructor: null, cone: null };
      continue;
    }
    if (!current) fail(`text before a query: ${line}`);
    if (line.startsWith("constructor ")) current.constructor = line.split(" ").slice(1);
    else if (line.startsWith("cone-report ")) current.cone = line.split(" ").slice(1);
    else if (line.startsWith("outcome ")) current.outcome = line.split(" ").slice(1);
    else if (line.startsWith("hit ")) current.hits.push(line.split(" ")[2]);
    else if (line.startsWith("label ") || line.startsWith("skips ")) continue;
    else fail(`unexpected line: ${line}`);
  }
  push();
  return blocks;
}

function sameHeader(left, right) {
  return left.length === right.length && left.every((token, index) => token === right[index]);
}

function format(value) {
  return value.toPrecision(12);
}

function metrics(order, relevant) {
  const scores = { ndcg: ndcg(order, relevant) };
  for (const k of CUTOFFS) {
    scores[`p${k}`] = hitsAt(order, relevant, k) / k;
    scores[`r${k}`] = hitsAt(order, relevant, k) / relevant.length;
  }
  return scores;
}

function statusOf(block) {
  if (block.constructor) return block.constructor[0];
  if (!block.outcome) return "missing";
  if (block.outcome[0] === "ranked" && block.outcome[4] === "29" && block.hits.length === 29) return "ranked";
  return block.outcome[0] === "ranked" ? "incomplete" : block.outcome[0];
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function verdictFor(systemRows, baselineRows) {
  if (systemRows.some((row) => row.status !== "ranked") || baselineRows.some((rows) => rows.some((row) => row.status !== "ranked"))) {
    return "blocked";
  }
  const systemMean = mean(systemRows.map((row) => row.scores.ndcg));
  const baselineMeans = baselineRows.map((rows) => mean(rows.map((row) => row.scores.ndcg)));
  if (baselineMeans.every((baseline) => systemMean > baseline)) return "positive";
  if (baselineMeans.every((baseline) => systemMean < baseline)) return "negative";
  return "inconclusive";
}

function score(population, listing) {
  const { snapshot, lemmas, queries } = population;
  const expected = expectedBlocks(queries);
  const parsed = parseInspect(listing);
  if (parsed.length !== expected.length) fail(`expected ${expected.length} inspector blocks, found ${parsed.length}`);
  const bySystem = { point: [], lexical: [], difference: [], shared: [] };
  const cones = [];
  expected.forEach((item, index) => {
    const block = parsed[index];
    if (!sameHeader(block.header, item.header)) {
      fail(`block ${index + 1} is ${block.header.join(" ")}, expected ${item.header.join(" ")}`);
    }
    if (item.system === "cone") {
      const search = block.outcome ? block.outcome[0] : block.constructor ? block.constructor[0] : "missing";
      const relevantInside = item.query.relevant_source_ids.filter((id) => block.hits.includes(id)).length;
      cones.push({
        query: item.query.query_id,
        theta: item.theta,
        search,
        diagnostic: block.cone ? block.cone[1] : "missing",
        inside: block.cone ? block.cone[5] : "missing",
        purity: block.cone ? block.cone[7] : "missing",
        relevantInside,
      });
      return;
    }
    const status = statusOf(block);
    const row = { query: item.query.query_id, status };
    if (status === "ranked") row.scores = metrics(block.hits, item.query.relevant_source_ids);
    bySystem[item.system].push(row);
  });
  for (const query of queries) {
    const order = lexicalOrder(query, lemmas, snapshot.abstract_corpus);
    if (order.length !== 29) fail(`lexical order for ${query.query_id} has ${order.length} entries`);
    bySystem.lexical.push({ query: query.query_id, status: "ranked", scores: metrics(order, query.relevant_source_ids) });
  }
  const pairIds = new Set(queries.filter((query) => query.concrete_source_ids.length === 2).map((query) => query.query_id));
  const pointPairs = bySystem.point.filter((row) => pairIds.has(row.query));
  const lexicalPairs = bySystem.lexical.filter((row) => pairIds.has(row.query));
  return { bySystem, cones, pointPairs, lexicalPairs };
}

function printRows(system, rows) {
  for (const row of rows) {
    if (row.status !== "ranked") {
      process.stdout.write(`query ${row.query} system ${system} status ${row.status}\n`);
      continue;
    }
    const scores = CUTOFFS.flatMap((k) => [`p${k} ${format(row.scores[`p${k}`])}`, `r${k} ${format(row.scores[`r${k}`])}`]).join(" ");
    process.stdout.write(`query ${row.query} system ${system} status ranked ndcg ${format(row.scores.ndcg)} ${scores}\n`);
  }
}

function printMean(system, rows) {
  const ranked = rows.filter((row) => row.status === "ranked");
  if (ranked.length === 0) {
    process.stdout.write(`mean ${system} ranked 0\n`);
    return;
  }
  process.stdout.write(`mean ${system} ranked ${ranked.length} ndcg ${format(mean(ranked.map((row) => row.scores.ndcg)))}\n`);
}

checkFixture();
const population = loadPopulation();
const writePath = process.argv.includes("--write-queries") ? argument("--write-queries") : "";
if (writePath) writeQueries(population.queries, writePath);
if (!process.argv.includes("--inspect")) {
  if (!writePath) fail("pass --write-queries or --inspect");
  process.stdout.write(`wrote ${population.queries.length} query groups\n`);
} else {
  const scored = score(population, readFileSync(argument("--inspect"), "utf8"));
  for (const system of ["point", "lexical", "shared", "difference"]) printRows(system, scored.bySystem[system]);
  for (const system of ["point", "lexical", "shared", "difference"]) printMean(system, scored.bySystem[system]);
  printMean("point-pairs", scored.pointPairs);
  printMean("lexical-pairs", scored.lexicalPairs);
  process.stdout.write(`verdict shared ${verdictFor(scored.bySystem.shared, [scored.bySystem.point, scored.bySystem.lexical])}\n`);
  process.stdout.write(`verdict difference ${verdictFor(scored.bySystem.difference, [scored.pointPairs, scored.lexicalPairs])}\n`);
  for (const cone of scored.cones) {
    process.stdout.write(`cone ${cone.query} theta ${cone.theta} search ${cone.search} diagnostic ${cone.diagnostic} inside ${cone.inside} relevant ${cone.relevantInside} purity ${cone.purity}\n`);
  }
}
