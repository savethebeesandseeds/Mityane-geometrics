// SPDX-License-Identifier: AGPL-3.0-only
// Fixed version-2 hierarchy: Hero -> Goal -> Milestone. Only milestones have states.
export const STATUS_LABELS = Object.freeze({proposed: 'Proposed', ready: 'Ready', in_progress: 'In progress', blocked: 'Blocked', done: 'Done'});
export const STATUS_ORDER = Object.freeze(['in_progress', 'blocked', 'ready', 'proposed', 'done']);
export const OUTCOME_LABELS = Object.freeze({supported: 'Supported', negative: 'Negative', inconclusive: 'Inconclusive', no_eligible_candidate: 'No eligible candidate', favorable: 'Favorable', eligible: 'Eligible', rejected: 'Rejected'});
const slug = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const insist = (condition, message) => { if (!condition) throw new Error(message); };
const text = (value, name) => insist(typeof value === 'string' && value.trim().length > 0, `${name} must be nonempty text`);
const id = (value, name) => insist(typeof value === 'string' && slug.test(value), `${name} must be a slug`);
function shape(value, required, optional, name) {
  insist(value !== null && typeof value === 'object' && !Array.isArray(value), `${name} must be an object`);
  insist(required.every(key => Object.hasOwn(value, key)), `${name} is missing required fields`);
  insist(Object.keys(value).every(key => [...required, ...optional].includes(key)), `${name} has unknown fields`);
}
function date(value, name) {
  insist(typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value), `${name} must be a YYYY-MM-DD date`);
  const timestamp = Date.parse(`${value}T00:00:00Z`);
  insist(Number.isFinite(timestamp) && new Date(timestamp).toISOString().slice(0, 10) === value, `${name} must be a real calendar date`);
}
const array = (value, name, maximum = 2000) => insist(Array.isArray(value) && value.length <= maximum, `${name} must be a bounded array`);
function strings(value, name) { array(value, name); value.forEach(entry => text(entry, name)); }
export function isSafeEvidencePath(value) {
  if (typeof value !== 'string' || value.length > 500) return false;
  const [file, anchor, ...extra] = value.split('#');
  return extra.length === 0 && (!anchor || /^[a-zA-Z0-9_-]+$/.test(anchor)) &&
    /^[a-zA-Z0-9_.-]+(?:\/[a-zA-Z0-9_.-]+)*$/.test(file) && file.split('/').every(part => part !== '.' && part !== '..');
}
export function validateManifest(value) {
  shape(value, ['schemaVersion', 'title', 'description', 'heroes'], ['focusMilestones'], 'Manifest');
  insist(value.schemaVersion === 2, 'Unsupported manifest schemaVersion');
  text(value.title, 'Manifest title'); text(value.description, 'Manifest description');
  if (Object.hasOwn(value, 'focusMilestones')) {
    strings(value.focusMilestones, 'Manifest focusMilestones');
    value.focusMilestones.forEach(entry => id(entry, 'Focus Milestone id'));
    insist(new Set(value.focusMilestones).size === value.focusMilestones.length, 'Duplicate focus Milestone');
  }
  array(value.heroes, 'Manifest heroes', 100);
  const ids = new Set(), files = new Set();
  for (const entry of value.heroes) {
    shape(entry, ['id', 'file'], [], 'Manifest Hero'); id(entry.id, 'Manifest Hero id');
    insist(typeof entry.file === 'string' && /^heroes\/[0-9]{2,3}-[a-z0-9-]+\.json$/.test(entry.file), 'Manifest Hero file must be a numbered JSON in heroes/');
    insist(!ids.has(entry.id) && !files.has(entry.file), 'Duplicate manifest Hero id or file');
    ids.add(entry.id); files.add(entry.file);
  }
  return value;
}
export function validateHero(hero) {
  shape(hero, ['schemaVersion', 'id', 'number', 'title', 'summary', 'updatedAt', 'goals'], ['purpose', 'boundary'], 'Hero');
  insist(hero.schemaVersion === 2, 'Unsupported Hero schemaVersion'); id(hero.id, 'Hero id');
  insist(Number.isInteger(hero.number) && hero.number > 0 && hero.number <= 100, 'Hero number must be an integer from 1 to 100');
  text(hero.title, 'Hero title'); text(hero.summary, 'Hero summary'); date(hero.updatedAt, 'Hero updatedAt');
  for (const key of ['purpose', 'boundary']) if (Object.hasOwn(hero, key)) text(hero[key], `Hero ${key}`);
  array(hero.goals, 'Hero goals', 200);
  const ids = new Set([hero.id]); let count = 0;
  const unique = value => { id(value, 'Goal or Milestone id'); insist(!ids.has(value), `Duplicate id: ${value}`); ids.add(value); };
  for (const goal of hero.goals) {
    shape(goal, ['id', 'title', 'milestones'], ['description', 'type'], 'Goal'); unique(goal.id); text(goal.title, 'Goal title');
    if (Object.hasOwn(goal, 'type')) insist(['outcome', 'ongoing', 'conditional'].includes(goal.type), 'Goal type must be outcome or ongoing or conditional');
    if (Object.hasOwn(goal, 'description')) text(goal.description, 'Goal description');
    array(goal.milestones, 'Goal milestones');
    for (const milestone of goal.milestones) {
      insist(++count <= 2000, 'Hero milestone count exceeds limit');
      shape(milestone, ['id', 'title', 'status'], ['description', 'owner', 'updatedAt', 'completedAt', 'acceptance', 'evidence', 'dependsOn', 'verification', 'outcome', 'requiresOutcomes', 'requiresCurrentEvidence'], 'Milestone');
      unique(milestone.id); text(milestone.title, 'Milestone title');
      insist(Object.hasOwn(STATUS_LABELS, milestone.status), `Unknown Milestone status: ${milestone.status}`);
      if (Object.hasOwn(milestone, 'description')) text(milestone.description, 'Milestone description');
      if (Object.hasOwn(milestone, 'owner')) insist(milestone.owner === null || (typeof milestone.owner === 'string' && milestone.owner.trim()), 'Milestone owner must be text or null');
      if (Object.hasOwn(milestone, 'updatedAt')) { date(milestone.updatedAt, 'Milestone updatedAt'); insist(milestone.updatedAt <= hero.updatedAt, 'Hero updatedAt precedes a Milestone'); }
      if (milestone.completedAt != null) {
        date(milestone.completedAt, 'Milestone completedAt');
        insist(milestone.status === 'done' && milestone.completedAt <= (milestone.updatedAt || hero.updatedAt), 'Completion date requires done status and cannot follow updatedAt');
      }
      if (Object.hasOwn(milestone, 'acceptance')) strings(milestone.acceptance, 'Milestone acceptance');
      if (Object.hasOwn(milestone, 'dependsOn')) {
        strings(milestone.dependsOn, 'Milestone dependsOn');
        insist(new Set(milestone.dependsOn).size === milestone.dependsOn.length && milestone.dependsOn.every(dependency => slug.test(dependency) && dependency !== milestone.id), 'Dependencies must be distinct other Milestone IDs');
      }
      if (Object.hasOwn(milestone, 'outcome')) {
        insist(Object.hasOwn(OUTCOME_LABELS, milestone.outcome), 'Unknown scientific outcome');
        insist(milestone.status === 'done', 'Scientific outcome requires done work status');
      }
      if (Object.hasOwn(milestone, 'verification')) {
        const verification = milestone.verification;
        shape(verification, ['status', 'scope', 'reason'], ['receipt', 'receiptSha256'], 'Verification');
        insist(['current', 'historical', 'unverified'].includes(verification.status), 'Unknown verification status');
        text(verification.scope, 'Verification scope'); text(verification.reason, 'Verification reason');
        if (Object.hasOwn(verification, 'receipt')) insist(isSafeEvidencePath(verification.receipt) && !verification.receipt.includes('#'), 'Unsafe verification receipt path');
        if (Object.hasOwn(verification, 'receiptSha256')) insist(typeof verification.receiptSha256 === 'string' && /^[a-f0-9]{64}$/.test(verification.receiptSha256), 'Verification receiptSha256 must be lowercase SHA256');
        if (verification.status === 'current') insist(verification.receipt && verification.receiptSha256, 'Current verification requires receipt and receiptSha256');
      }
      if (Object.hasOwn(milestone, 'requiresCurrentEvidence')) {
        strings(milestone.requiresCurrentEvidence, 'Milestone requiresCurrentEvidence');
        insist(new Set(milestone.requiresCurrentEvidence).size === milestone.requiresCurrentEvidence.length, 'Repeated current evidence requirement');
        milestone.requiresCurrentEvidence.forEach(dependency => insist((milestone.dependsOn || []).includes(dependency), 'Current evidence requirement must be a direct dependency'));
      }
      if (Object.hasOwn(milestone, 'requiresOutcomes')) {
        array(milestone.requiresOutcomes, 'Milestone requiresOutcomes');
        const required = new Set();
        for (const requirement of milestone.requiresOutcomes) {
          shape(requirement, ['milestone', 'outcomes'], [], 'Outcome requirement');
          insist((milestone.dependsOn || []).includes(requirement.milestone) && !required.has(requirement.milestone), 'Outcome requirement must identify a distinct direct dependency');
          required.add(requirement.milestone); strings(requirement.outcomes, 'Allowed outcomes');
          insist(requirement.outcomes.length > 0 && new Set(requirement.outcomes).size === requirement.outcomes.length && requirement.outcomes.every(outcome => Object.hasOwn(OUTCOME_LABELS, outcome)), 'Allowed outcomes must be distinct known outcomes');
        }
      }
      if (Object.hasOwn(milestone, 'evidence')) {
        array(milestone.evidence, 'Milestone evidence');
        for (const entry of milestone.evidence) {
          shape(entry, ['label', 'path'], [], 'Evidence'); text(entry.label, 'Evidence label');
          insist(isSafeEvidencePath(entry.path), `Unsafe evidence path: ${entry.path}`);
        }
      }
      if (milestone.status === 'done') insist(milestone.acceptance?.length > 0 && milestone.evidence?.length > 0, `Done Milestone ${milestone.id} requires acceptance criteria and evidence`);
      if (milestone.status === 'blocked') text(milestone.description, 'Blocked Milestone reason');
    }
  }
  return hero;
}
export function flattenMilestones(heroes) {
  return heroes.flatMap(hero => hero.goals.flatMap(goal => goal.milestones.map(milestone => ({milestone, hero, goal}))));
}
export function validateProject(manifest, heroes) {
  validateManifest(manifest); insist(heroes.length === manifest.heroes.length, 'Project Hero set differs from manifest');
  const numbers = new Set(), identifiers = new Set(), milestones = new Map();
  const unique = value => { insist(!identifiers.has(value), `Duplicate global id: ${value}`); identifiers.add(value); };
  heroes.forEach((hero, index) => {
    validateHero(hero); insist(hero.id === manifest.heroes[index].id, 'Hero ID or order differs from manifest');
    insist(!numbers.has(hero.number), 'Duplicate Hero number'); numbers.add(hero.number); unique(hero.id);
    for (const goal of hero.goals) { unique(goal.id); for (const milestone of goal.milestones) { unique(milestone.id); milestones.set(milestone.id, milestone); } }
  });
  for (const milestone of milestones.values()) for (const dependency of milestone.dependsOn || []) {
    insist(milestones.has(dependency), `Missing Milestone dependency ${dependency} for ${milestone.id}`);
    if (['done', 'ready'].includes(milestone.status)) insist(milestones.get(dependency).status === 'done', `${milestone.status === 'done' ? 'Done' : 'Ready'} Milestone ${milestone.id} depends on unfinished Milestone ${dependency}`);
  }
  for (const focus of manifest.focusMilestones || []) insist(milestones.has(focus), `Missing focus Milestone ${focus}`);
  const visiting = new Set(), visited = new Set();
  function visit(id) {
    insist(!visiting.has(id), `Dependency cycle at ${id}`); if (visited.has(id)) return;
    visiting.add(id); (milestones.get(id).dependsOn || []).forEach(visit); visiting.delete(id); visited.add(id);
  }
  milestones.forEach((_, id) => visit(id));
}
// Completion is historical work accounting. Advancement additionally checks
// explicit scientific outcomes and evidence applicability throughout ancestry.
export function verificationAssessment(milestone, assessments = {}) {
  const authored = milestone.verification;
  if (!authored) return {status: 'unverified', scope: 'Not assessed', reason: 'No current evidence assessment is recorded.'};
  const assessed = assessments[milestone.id];
  if (assessed && ['current', 'stale', 'historical', 'unverified'].includes(assessed.status)) return assessed;
  return {status: authored.status === 'historical' ? 'historical' : 'unverified', scope: authored.scope,
    reason: authored.status === 'historical' ? authored.reason : 'Current repository bytes have not been checked against the pinned receipt.'};
}
function readinessIndex(heroes, assessments) {
  const index = new Map(flattenMilestones(heroes).map(entry => [entry.milestone.id, entry]));
  const cache = new Map(), active = new Set();
  function assess(milestone) {
    if (cache.has(milestone.id)) return cache.get(milestone.id);
    if (active.has(milestone.id)) return {ready: false, complete: milestone.status === 'done', blockers: [{id: milestone.id, reason: 'Dependency cycle prevents readiness.'}]};
    active.add(milestone.id);
    const blockers = [], add = (id, reason) => {
      if (!blockers.some(entry => entry.id === id && entry.reason === reason)) blockers.push({id, reason});
    };
    if (milestone.status === 'blocked') add(milestone.id, milestone.description || 'Work is explicitly blocked.');
    for (const dependencyId of milestone.dependsOn || []) {
      const dependency = index.get(dependencyId)?.milestone;
      if (!dependency) { add(dependencyId, 'Required milestone is missing.'); continue; }
      if (dependency.status !== 'done') add(dependencyId, `Required work is ${STATUS_LABELS[dependency.status] || 'unfinished'}.`);
      // Gates on completed ancestors still apply to current advancement.
      assess(dependency).blockers.forEach(blocker => add(blocker.id, blocker.reason));
    }
    for (const requirement of milestone.requiresOutcomes || []) {
      const dependency = index.get(requirement.milestone)?.milestone;
      if (dependency?.status !== 'done' || !requirement.outcomes.includes(dependency.outcome)) {
        add(requirement.milestone, `Requires ${requirement.outcomes.map(outcome => OUTCOME_LABELS[outcome] || outcome).join(' or ')}; recorded outcome is ${OUTCOME_LABELS[dependency?.outcome] || 'not recorded'}.`);
      }
    }
    for (const dependencyId of milestone.requiresCurrentEvidence || []) {
      const dependency = index.get(dependencyId)?.milestone;
      const verification = dependency ? verificationAssessment(dependency, assessments) : {status: 'unverified', reason: 'Required milestone is missing.'};
      if (verification.status !== 'current') add(dependencyId, `Current evidence required: ${verification.status}. ${verification.reason}`);
    }
    active.delete(milestone.id);
    const result = {ready: milestone.status !== 'done' && blockers.length === 0, complete: milestone.status === 'done', blockers};
    cache.set(milestone.id, result); return result;
  }
  return {index, assess};
}
export function validateReadyMilestones(heroes, verificationAssessments = {}) {
  const {assess} = readinessIndex(heroes, verificationAssessments);
  for (const {milestone} of flattenMilestones(heroes)) {
    if (milestone.status !== 'ready') continue;
    const {blockers} = assess(milestone);
    insist(blockers.length === 0, `Ready Milestone ${milestone.id} has unmet advancement requirements: ${blockers.map(blocker => blocker.reason).join(' ')}`);
  }
}

export function milestoneReadiness(milestone, heroes, verificationAssessments = {}) {
  return readinessIndex(heroes, verificationAssessments).assess(milestone);
}
export function deriveCurrentCampaign(heroes, focusMilestones, assessments = {}) {
  const {index, assess} = readinessIndex(heroes, assessments);
  const entry = record => ({...record, readiness: assess(record.milestone)});
  const focus = (focusMilestones || []).map(identifier => index.get(identifier)).filter(Boolean);
  const selected = focus.find(record => record.milestone.status !== 'done') || null;
  const committedIds = new Set();
  function include(identifier) {
    if (committedIds.has(identifier)) return;
    const record = index.get(identifier); if (!record) return;
    committedIds.add(identifier); (record.milestone.dependsOn || []).forEach(include);
  }
  focus.forEach(record => include(record.milestone.id));
  const committed = [...index.values()].filter(record => committedIds.has(record.milestone.id) && record.goal.type !== 'conditional').map(entry);
  const blockers = selected ? assess(selected.milestone).blockers : [];
  // Failed scientific/evidence gates are not executable actions. A separate
  // independent dependency may still be ready, so do not stop the whole stream.
  const nextActions = committed.filter(record => record.readiness.ready && ['proposed', 'ready', 'in_progress'].includes(record.milestone.status)).slice(0, 3);
  const goals = heroes.flatMap(hero => hero.goals.map(goal => ({hero, goal})));
  return {decision: selected ? entry(selected) : null, committed, blockers, nextActions,
    ongoing: goals.filter(record => record.goal.type === 'ongoing'),
    conditional: goals.filter(record => record.goal.type === 'conditional')};
}
function countMilestones(milestones) {
  const counts = Object.fromEntries(Object.keys(STATUS_LABELS).map(status => [status, 0]));
  milestones.forEach(milestone => counts[milestone.status]++);
  return {total: milestones.length, done: counts.done, active: counts.in_progress, blocked: counts.blocked, ready: counts.ready, proposed: counts.proposed, counts};
}
export const summarize = heroes => countMilestones(flattenMilestones(heroes).map(({milestone}) => milestone));
export const summarizeGoal = goal => countMilestones(goal.milestones);
function rollup({total, done, active, blocked, ready}) {
  if (total > 0 && done === total) return 'done';
  if (blocked > 0) return 'blocked';
  if (active > 0 || done > 0) return 'in_progress';
  return ready > 0 ? 'ready' : 'proposed';
}
export const deriveHeroStatus = hero => rollup(summarize([hero]));
export const deriveGoalStatus = goal => rollup(summarizeGoal(goal));
export const heroLabel = hero => `Hx${String(hero.number).padStart(2, '0')}`;
export function filterHeroes(heroes, {query = '', status = 'all', heroId = 'all'} = {}) {
  const needle = query.trim().toLocaleLowerCase().replaceAll('ₓ', 'x');
  const contains = values => values.filter(Boolean).join(' ').toLocaleLowerCase().includes(needle);
  return heroes.flatMap(hero => {
    if (heroId !== 'all' && hero.id !== heroId) return [];
    const heroMatch = contains([heroLabel(hero), hero.id, hero.title, hero.summary, hero.purpose, hero.boundary]);
    const goals = hero.goals.flatMap(goal => {
      const goalMatch = heroMatch || contains([goal.id, goal.title, goal.description]);
      const milestones = goal.milestones.filter(milestone => (status === 'all' || milestone.status === status) &&
        (goalMatch || contains([milestone.id, milestone.title, milestone.description, milestone.owner, ...(milestone.acceptance || []), ...(milestone.evidence || []).flatMap(entry => [entry.label, entry.path])])));
      const emptyPlanningGoal = goal.milestones.length === 0 && goalMatch && status === 'all';
      return milestones.length || emptyPlanningGoal ? [{...goal, milestones}] : [];
    });
    const emptyPlanningHero = hero.goals.length === 0 && heroMatch && status === 'all';
    return goals.length || emptyPlanningHero ? [{...hero, goals}] : [];
  });
}
export async function loadHeroFiles(files) {
  const selected = Array.from(files || []), errors = [];
  insist(selected.length > 0, 'Select the numbered Hero JSON files from web/dashboard/heroes.');
  insist(selected.length <= 100, 'Select at most 100 Hero JSON files.');
  const results = await Promise.allSettled(selected.map(async file => {
    insist(typeof file.name === 'string' && /\.json$/i.test(file.name), 'Select a Hero JSON file.');
    insist(typeof file.text === 'function', 'The selected file cannot be read.');
    insist(file.size == null || (Number.isFinite(file.size) && file.size >= 0 && file.size <= 4 * 1024 * 1024), 'Hero JSON exceeds the 4 MiB file limit.');
    const contents = await file.text();
    insist(typeof contents === 'string' && contents.length <= 4 * 1024 * 1024, 'Hero JSON exceeds the text limit.');
    return validateHero(JSON.parse(contents.replace(/^\uFEFF/, '')));
  }));
  const heroes = [];
  results.forEach((result, index) => {
    if (result.status === 'fulfilled') heroes.push(result.value);
    else errors.push({file: selected[index].name || 'Selected file', message: String(result.reason.message || result.reason)});
  });
  insist(heroes.length > 0, `No valid Hero JSON files were selected. ${errors.map(error => `${error.file}: ${error.message}`).join(' ')}`);
  heroes.sort((left, right) => left.number - right.number);
  const manifest = {
    schemaVersion: 2, title: 'Mityane geometrics project dashboard', description: 'Heroes loaded from explicitly selected local JSON files.',
    heroes: heroes.map(hero => ({id: hero.id, file: `heroes/${String(hero.number).padStart(2, '0')}-${hero.id}.json`})),
  };
  try { validateProject(manifest, heroes); validateReadyMilestones(heroes); } catch (error) { errors.push({file: 'Selected Heroes', message: error.message}); }
  return {manifest, heroes, errors};
}
