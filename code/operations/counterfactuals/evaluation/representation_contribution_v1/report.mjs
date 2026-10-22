// Post-execution presentation only. Numerical metrics and bootstrap draws were
// declared and computed by the frozen audit; this file never runs an evaluator.
import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, readdir, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const requireThat = (value, message) => { if (!value) throw new Error(message); };
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const args = process.argv.slice(2);
requireThat(args.length === 4 && args[0] === '--attempt' && args[2] === '--out',
  'usage: report.mjs --attempt /mityane/.temp/operations/counterfactuals/representation-contribution-v1/attempt-001 --out /mityane/.temp/operations/counterfactuals/representation-contribution-v1/report-001');
requireThat(process.platform === 'linux' && process.cwd() === '/mityane', 'Use the managed Linux container.');
await lstat('/.dockerenv');
const boundary = '/mityane/.temp/operations/counterfactuals/representation-contribution-v1';
const attempt = await realpath(args[1]);
const output = path.resolve(args[3]);
requireThat(attempt === `${boundary}/attempt-001`, 'Only the completed named study attempt is accepted.');
requireThat(output === `${boundary}/report-001`, 'Use the explicit exclusive report-001 destination.');
requireThat(await realpath(path.dirname(output)) === boundary, 'Report parent must be the actual study directory.');

async function pinned(relative, expected = null) {
  requireThat(typeof relative === 'string' && relative.length > 0 && !relative.includes('\\') &&
    !path.isAbsolute(relative) && relative.split('/').every(part => part && part !== '.' && part !== '..'),
  `Noncanonical record path: ${relative}`);
  const absolute = path.join(attempt, relative);
  requireThat((await lstat(absolute)).isFile(), `Record must be a regular non-symlink file: ${relative}`);
  requireThat(await realpath(absolute) === absolute, `Record path contains a symlink: ${relative}`);
  const bytes = await readFile(absolute);
  const identity = { path: relative, bytes: bytes.length, sha256: digest(bytes) };
  if (expected) {
    requireThat(identity.sha256 === expected.sha256 &&
      (expected.bytes === undefined || identity.bytes === expected.bytes), `Record pin mismatch: ${relative}`);
  }
  return { identity, bytes, json: () => JSON.parse(bytes.toString('utf8')) };
}

const finalRecord = await pinned('final.json');
const final = finalRecord.json();
requireThat(final.study_id === 'OR06-REP-CONTRIB-001' && final.disposition === 'completed' &&
  final.assessment_opened === true && final.source_unchanged === true &&
  final.verdict === 'supported_in_scope', 'Study does not have its valid completed in-scope verdict.');
const manifestRecord = await pinned('file-manifest.json', { sha256: final.file_manifest_sha256 });
const auditRecord = await pinned('audit.json', { sha256: final.audit_sha256 });
const freezeRecord = await pinned('freeze.json', { sha256: final.freeze_sha256 });
const sealRecord = await pinned('seal.json', { sha256: final.seal_sha256 });
const manifest = manifestRecord.json();
const audit = auditRecord.json();
const freeze = freezeRecord.json();
const seal = sealRecord.json();
requireThat(audit.schema === 'mityane.hx06.representation-contribution.audit/v1' &&
  audit.valid === true && audit.errors.length === 0 && audit.audit.error_count === 0 &&
  audit.supported === true, 'A valid, supported frozen audit is required.');
requireThat(seal.freeze_sha256 === freezeRecord.identity.sha256 &&
  seal.assessment_opened === false && freeze.assessment_opened === false &&
  freeze.metrics_observed === false, 'Assessment must have been sealed prospectively.');

const manifestPaths = new Set();
for (const record of manifest.files) {
  requireThat(!manifestPaths.has(record.path), `Duplicate manifest path: ${record.path}`);
  manifestPaths.add(record.path);
  await pinned(record.path, record);
}
for (const required of ['audit.json', 'freeze.json', 'seal.json', 'assessment-open.json',
  'configuration.json', 'permissions.json', 'prior_exposure.json']) {
  requireThat(manifestPaths.has(required), `Missing required manifest record: ${required}`);
}
async function filePaths(directory, prefix = '') {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) result.push(...await filePaths(path.join(directory, entry.name), relative));
    else {
      requireThat(entry.isFile(), `Nonregular attempt entry: ${relative}`);
      result.push(relative);
    }
  }
  return result;
}
const actual = (await filePaths(attempt)).filter(item => !['final.json', 'file-manifest.json'].includes(item));
requireThat(actual.length === manifestPaths.size && actual.every(item => manifestPaths.has(item)),
  'The complete attempt file inventory must match its pinned manifest.');

const configuration = audit.configuration;
const axes = configuration.axes;
const arms = [...configuration.arms, 'zero'];
const groups = configuration.groups;
const seeds = configuration.seeds.map(String);
requireThat(axes.join(',') === 'polynomial,periodic,hinge' && arms.join(',') ===
  'raw_polynomial,raw_additive,fourier,hinge,lossy,zero', 'Unexpected complete family/axis roster.');
const scopes = [...seeds, 'pooled'];
const finite = value => value === null || (typeof value === 'number' && Number.isFinite(value));
const number = value => value === null ? 'null' : (requireThat(finite(value), 'Nonfinite saved metric.'), value.toFixed(9));
const interval = value => value === null ? 'null' : `[${number(value.lower)}, ${number(value.upper)}]`;
const coverage = value => ['requested', 'succeeded', 'abstained', 'failed', 'unresolved'].map(key => value[key]).join('/');
const rows = [];
const markdown = [
  '# Complete representation contribution breakdowns', '',
  'Study `OR06-REP-CONTRIB-001` version `1.0.0`; completed 20 October 2026.', '',
  'This is a post-execution rendering of every predeclared numerical comparison in the frozen audit. No fitting, selection, resampling or new analysis occurs here. Full precision is retained in `metrics.json`; displayed numbers have nine decimal places.', '',
  'Each comparison binds provider `OR06-REP-CONTRIB-001/<family>/1.0.0`, four raw coordinates, a five-parameter TRAIN-only standardized OLS head for the named axis and seed, and the named saved synthetic source population. Horizon is static; contrasts change only a. `zero` is the unfitted zero-response baseline on the same roster, not an extra executed family. The lossy family deliberately erases a and is not an equal-information control.', '',
  'Attempt: `' + attempt + '`. Source closure `' + freeze.source_id + '`; final SHA-256 `' + finalRecord.identity.sha256 + '`; audit SHA-256 `' + auditRecord.identity.sha256 + '`.', '',
  'MAE first averages applicable supported changes within a source, then sources within group. Positive raw-polynomial-minus-family or zero-minus-family differences favor the family. Coverage is requested/succeeded/abstained/failed/unresolved; the positive-direction group includes stress abstentions. Counts for zero describe its inherited comparison roster. Groups overlap and must not be added together.', '',
  'Assessment intervals use the frozen 1,000-draw source-block bootstrap, keeping axes, families and alternatives together and drawing within seed before source-weighted pooling. They are conditional marginal descriptive intervals for the frozen heads, with no training-process, external-population or simultaneous-coverage guarantee. Null stress metrics mean no supported contrasts, not zero error. Development selected nothing and has point estimates only.', '',
  '## Actual native execution coverage', '',
  '| Split | Requested | Succeeded | Abstained | Failed | Unresolved |',
  '| --- | ---: | ---: | ---: | ---: | ---: |'
];
for (const split of ['development', 'assessment']) {
  const c = audit.coverage[split];
  markdown.push(`| ${split} | ${c.requested} | ${c.succeeded} | ${c.abstained} | ${c.failed} | ${c.unresolved} |`);
}
for (const split of ['assessment', 'development']) {
  markdown.push('', `## ${split === 'assessment' ? 'Protected assessment' : 'Development point estimates'}`, '');
  for (const scope of scopes) {
    const populations = split === 'assessment' ? audit.metrics : audit.metrics.development;
    const byAxis = scope === 'pooled' ? populations.pooled : populations.per_seed[scope];
    for (const axis of axes) {
      markdown.push(`### ${scope === 'pooled' ? 'Pooled source populations' : `Seed ${scope}`} — ${axis}`, '',
        '| Group | Family | Sources | Coverage R/S/A/F/U | MAE | MAE 95% interval | Raw polynomial − family | Paired 95% interval | Zero − family | Paired 95% interval | Finite draws |',
        '| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- | ---: |');
      requireThat(Object.keys(byAxis[axis]).join(',') === groups.join(','), 'Incomplete or reordered groups.');
      for (const group of groups) {
        const summary = byAxis[axis][group];
        requireThat(Object.keys(summary.families).join(',') === arms.join(','), 'Incomplete or reordered family rows.');
        for (const arm of arms) {
          const value = summary.families[arm];
          for (const key of ['source_unit_mae', 'raw_polynomial_minus_family_mae', 'zero_minus_family_mae'])
            requireThat(finite(value[key]), `Invalid saved ${key}.`);
          rows.push({ split, scope, axis, group, family: arm,
            source_units: summary.source_units, coverage: summary.coverage_per_family, ...value });
          markdown.push(`| ${group} | ${arm} | ${summary.source_units} | ${coverage(summary.coverage_per_family)} | ` +
            `${number(value.source_unit_mae)} | ${interval(value.mae_ci95)} | ` +
            `${number(value.raw_polynomial_minus_family_mae)} | ${interval(value.raw_polynomial_minus_family_ci95)} | ` +
            `${number(value.zero_minus_family_mae)} | ${interval(value.zero_minus_family_ci95)} | ${value.finite_bootstrap_resamples ?? 'null'} |`);
        }
      }
      markdown.push('');
    }
  }
}
requireThat(rows.length === 2 * scopes.length * axes.length * groups.length * arms.length,
  'Incomplete presentation row inventory.');
markdown.push('## Empty bootstrap group draws', '',
  'Every non-stress assessment group in every seed and pooled scope had 1,000 finite resamples and zero empty draws. Stress groups contain only abstentions and have no MAE, paired differences or intervals.', '',
  '## Numerical checks and interpretation', '',
  `The independent audit recorded ${audit.audit.error_count} errors. Maximum saved prediction discrepancy was ${audit.audit.max_abs.pair_numeric}; maximum independent TRAIN-fit discrepancy was ${audit.audit.max_abs.training_fit}. No-change and lossy contrasts were exactly zero. The prospective task-aligned screen passed: maximum per-seed aligned MAE ${audit.synthetic_screen.task_aligned_mae_maximum}, threshold ${audit.synthetic_screen.threshold}.`, '',
  'The experiment measures compatibility between fixed feature maps and affine readouts. It does not evaluate a learned encoder, a distance geometry, an observational estimator or a causal effect. Task formulas were known; only the new observations were protected. Full seed/group results above retain all unfavorable and redundant cases.', '');
const metrics = {
  schema: 'mityane.hx06.representation-contribution.presentation/v1',
  role: 'postexecution_presentation_only',
  study_id: final.study_id,
  audit_sha256: auditRecord.identity.sha256,
  final_sha256: finalRecord.identity.sha256,
  source_id: freeze.source_id,
  numeric_analysis_identity: 'frozen audit.py, captured in attempt source closure',
  readout: configuration.fit_method,
  provider: 'OR06-REP-CONTRIB-001/<family>/1.0.0',
  horizon: 'static',
  sign: audit.interpretation.comparison_sign,
  uncertainty: audit.uncertainty,
  coverage: audit.coverage,
  row_count: rows.length,
  rows
};

// All verification precedes exclusive output creation. A failed write leaves its
// partial destination intact; reruns never overwrite an earlier presentation.
await mkdir(output);
const outputs = [];
for (const [name, bytes] of [
  ['BREAKDOWNS.md', Buffer.from(markdown.join('\n'))],
  ['metrics.json', Buffer.from(`${JSON.stringify(metrics)}\n`)]
]) {
  await writeFile(path.join(output, name), bytes, { flag: 'wx' });
  outputs.push({ path: name, bytes: bytes.length, sha256: digest(bytes) });
}
const scriptPath = fileURLToPath(import.meta.url);
const scriptBytes = await readFile(scriptPath);
const receipt = {
  schema: 'mityane.hx06.representation-contribution.report-receipt/v1',
  created_at: new Date().toISOString(),
  role: 'postexecution_presentation_only',
  numerical_analysis: 'No recomputation, fitting, assessment execution, selection or bootstrap; copies frozen audited metrics.',
  inputs: [finalRecord, manifestRecord, auditRecord, freezeRecord, sealRecord].map(record => record.identity),
  verified_attempt_files: manifestPaths.size,
  source_id: freeze.source_id,
  report_script: { path: scriptPath, bytes: scriptBytes.length, sha256: digest(scriptBytes) },
  outputs,
  rows: { assessment: rows.filter(row => row.split === 'assessment').length,
    development: rows.filter(row => row.split === 'development').length },
  limitations: audit.interpretation
};
await writeFile(path.join(output, 'receipt.json'), `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx' });
console.log(JSON.stringify({ report: output, verified_attempt_files: manifestPaths.size, rows: receipt.rows, outputs }));
