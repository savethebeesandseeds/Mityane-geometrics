// Independent saved-record auditor for OR06-SYNTH-RESP-001.
// Deliberately imports no native feature, fitting, generator or metric helper.
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';

const INPUT_HEADER = ['unit', 'a_num', 'b_num', 'noise_sign'];
const MODEL_HEADER = ['arm', 'feature', 'mean', 'scale', 'coefficient'];
const PAIR_HEADER = ['arm', 'unit', 'change_num', 'a_num', 'b_num', 'alt_a_num',
  'state', 'reason', 'recorded_base', 'recorded_alt', 'delta', 'z_base', 'z_alt'];
const FIXED = {
  seeds: [1103, 2207, 3301],
  counts: { training: 512, development: 128, assessment: 256 },
  denominator: 4096,
  changes: [-1024, -512, 0, 512, 1024, 8192],
  arms: ['mixed', 'raw_polynomial', 'raw_linear', 'lossy'],
  bootstrap: { samples: 2000, seed: 9001 },
  thresholds: { numeric: 1e-10, mixed_mae: 0.02, equivalence_mae: 1e-9, zero_ratio: 0.2 },
  noise_denominator: 32,
};
const FEATURES = { mixed: 4, raw_polynomial: 4, raw_linear: 2, lossy: 4 };
const SPLITS = ['training', 'development', 'assessment'];
const SOURCE_FILES = { training: 'train.tsv', development: 'development.tsv', assessment: 'assessment.tsv' };
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const mean = (numbers) => numbers.reduce((a, b) => a + b, 0) / numbers.length;
function sameBinding(left, right) {
  if (Array.isArray(right)) return Array.isArray(left) && left.length === right.length && right.every((value, i) => sameBinding(left[i], value));
  if (right && typeof right === 'object') return left && typeof left === 'object'
    && Object.keys(left).length === Object.keys(right).length
    && Object.entries(right).every(([key, value]) => sameBinding(left[key], value));
  return left === right;
}

function lcg(seed) {
  let state = seed >>> 0;
  return () => (state = (Math.imul(1664525, state) + 1013904223) >>> 0);
}

// These are raw source-level formulas, separately transcribed from INTERFACE.
function features(arm, a, b) {
  switch (arm) {
    case 'mixed': return [a + b, a - b, a * b + a * a, a * b - a * a];
    case 'raw_polynomial': return [a, b, a * b, a * a];
    case 'raw_linear': return [a, b];
    case 'lossy': return [b, b * b, b * b * b, b * b * b * b];
    default: throw new Error(`Unknown arm ${arm}`);
  }
}

function reference(a, b) { return 1 + 2 * a - b + 1.5 * a * b + 0.5 * a * a; }
function referenceContrast(a, b, h) { return h * (2 + 1.5 * b + a) + 0.5 * h * h; }
function predict(model, raw) {
  let answer = model.intercept;
  for (let j = 0; j < raw.length; j++) {
    answer += model.coefficients[j] * (raw[j] - model.means[j]) / model.scales[j];
  }
  return answer;
}

function solve(matrix, right) {
  const n = right.length;
  const rows = matrix.map((row, i) => [...row, right[i]]);
  for (let k = 0; k < n; k++) {
    let pivot = k;
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(rows[i][k]) > Math.abs(rows[pivot][k])) pivot = i;
    }
    if (!(Math.abs(rows[pivot][k]) >= 1e-12)) throw new Error('Independent training fit is singular');
    [rows[k], rows[pivot]] = [rows[pivot], rows[k]];
    for (let i = k + 1; i < n; i++) {
      const multiplier = rows[i][k] / rows[k][k];
      for (let j = k; j <= n; j++) rows[i][j] -= multiplier * rows[k][j];
    }
  }
  const result = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let value = rows[i][n];
    for (let j = i + 1; j < n; j++) value -= rows[i][j] * result[j];
    result[i] = value / rows[i][i];
  }
  if (!result.every(Number.isFinite)) throw new Error('Independent training fit was nonfinite');
  return result;
}

function independentlyFit(training, arm, config) {
  const raw = training.map((row) => features(arm, row.a_num / config.denominator, row.b_num / config.denominator));
  const n = raw.length;
  const d = FEATURES[arm];
  const means = Array.from({ length: d }, (_, j) => mean(raw.map((row) => row[j])));
  const scales = means.map((m, j) => Math.sqrt(mean(raw.map((row) => (row[j] - m) ** 2))));
  if (!scales.every((x) => Number.isFinite(x) && x > 0)) throw new Error('Invalid independent training scale');
  const design = raw.map((row) => [1, ...row.map((value, j) => (value - means[j]) / scales[j])]);
  const gram = Array.from({ length: d + 1 }, () => Array(d + 1).fill(0));
  const right = Array(d + 1).fill(0);
  for (let i = 0; i < n; i++) {
    const source = training[i];
    const label = reference(source.a_num / config.denominator, source.b_num / config.denominator)
      + source.noise_sign / config.noise_denominator;
    for (let j = 0; j <= d; j++) {
      right[j] += design[i][j] * label;
      for (let k = 0; k <= d; k++) gram[j][k] += design[i][j] * design[i][k];
    }
  }
  const coefficients = solve(gram, right);
  return { means, scales, intercept: coefficients[0], coefficients: coefficients.slice(1), gram, right };
}

function regenerate(seed, config) {
  const next = lcg(seed);
  const usedB = new Set();
  const result = {};
  for (const split of SPLITS) {
    result[split] = [];
    while (result[split].length < config.counts[split]) {
      const a_num = (next() % 6145) - 3072;
      const b_num = (next() % 8193) - 4096;
      if (usedB.has(b_num)) continue;
      usedB.add(b_num);
      const noise_sign = (next() >>> 31) === 1 ? 1 : -1;
      result[split].push({ unit: `${split}-${result[split].length}`, a_num, b_num, noise_sign });
    }
  }
  return result;
}

function parseTable(bytes, header, label, allowEmpty = []) {
  const text = bytes.toString('utf8');
  if (!Buffer.from(text, 'utf8').equals(bytes)) throw new Error(`${label}: invalid UTF-8`);
  if (text.includes('\r') || text.includes('\0')) throw new Error(`${label}: unexpected CR or NUL`);
  const lines = text.endsWith('\n') ? text.slice(0, -1).split('\n') : text.split('\n');
  if (lines.shift() !== header.join('\t')) throw new Error(`${label}: wrong TSV header`);
  return lines.map((line, i) => {
    const values = line.split('\t');
    if (values.length !== header.length || values.some((value, j) => value === '' && !allowEmpty.includes(header[j]))) {
      throw new Error(`${label}:${i + 2}: malformed TSV row`);
    }
    return Object.fromEntries(header.map((key, j) => [key, values[j]]));
  });
}

function integer(text, label) {
  if (!/^-?(0|[1-9]\d*)$/.test(text) || !Number.isSafeInteger(Number(text))) {
    throw new Error(`${label}: invalid integer`);
  }
  return Number(text);
}

function numeric(text, label) {
  if (!/^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(text) || !Number.isFinite(Number(text))) {
    throw new Error(`${label}: invalid finite number`);
  }
  return Number(text);
}

function hexText(hex, label) {
  if (!/^(?:[0-9a-f]{2})*$/.test(hex)) throw new Error(`${label}: invalid hex text`);
  const bytes = Buffer.from(hex, 'hex');
  const value = bytes.toString('utf8');
  if (!Buffer.from(value, 'utf8').equals(bytes)) throw new Error(`${label}: invalid UTF-8 payload`);
  return value;
}

// The complete typed roster has hundreds of thousands of fields. Yield rows
// without retaining a second array of decoded request records in memory.
function* requestFields(bytes, label) {
  const text = bytes.toString('utf8');
  if (!Buffer.from(text, 'utf8').equals(bytes) || text.includes('\r') || text.includes('\0')) {
    throw new Error(`${label}: invalid request encoding`);
  }
  const header = 'arm\tpair_id\tscope\tfield\tvalue_hex\n';
  if (!text.startsWith(header)) throw new Error(`${label}: wrong request header`);
  let start = header.length;
  while (start < text.length) {
    const end = text.indexOf('\n', start);
    const line = text.slice(start, end < 0 ? text.length : end);
    const values = line.split('\t');
    if (values.length !== 5 || values.slice(0, 4).some((value) => value === '')) {
      throw new Error(`${label}: malformed request row`);
    }
    yield { arm: values[0], pair_id: values[1], scope: values[2], field: values[3], value_hex: values[4] };
    start = end < 0 ? text.length : end + 1;
  }
}

function summary(unitErrors, coverage, maxima) {
  const scores = {};
  for (const [arm, units] of Object.entries(unitErrors)) scores[arm] = mean(units);
  return { source_units: unitErrors.zero.length, source_unit_mae: scores, coverage, ...maxima };
}

function bootstrap(seeds, config) {
  const next = lcg(config.bootstrap.seed);
  const arms = [...config.arms, 'zero'];
  const samples = Object.fromEntries(arms.map((arm) => [arm, []]));
  const differences = [];
  const ratios = [];
  for (let r = 0; r < config.bootstrap.samples; r++) {
    const value = Object.fromEntries(arms.map((arm) => [arm, 0]));
    for (const seed of seeds) {
      const n = seed.zero.length;
      const subtotal = Object.fromEntries(arms.map((arm) => [arm, 0]));
      for (let i = 0; i < n; i++) {
        // Full-range mapping avoids the LCG's short low-bit period at n=256.
        const index = Math.floor(next() / 4294967296 * n);
        for (const arm of arms) subtotal[arm] += seed[arm][index];
      }
      for (const arm of arms) value[arm] += subtotal[arm] / n / seeds.length;
    }
    for (const arm of arms) samples[arm].push(value[arm]);
    differences.push(value.mixed - value.raw_polynomial);
    ratios.push(value.mixed / value.zero);
  }
  const interval = (values) => {
    values.sort((a, b) => a - b);
    return { lower: values[Math.floor(0.025 * values.length)], upper: values[Math.floor(0.975 * values.length)] };
  };
  return {
    method: 'source-unit bootstrap within assessment seed; macro-average frozen-model MAEs',
    samples: config.bootstrap.samples,
    generator: 'unsigned 32-bit LCG 1664525/1013904223',
    index_mapping: 'floor(next / 4294967296 * source_unit_count); draw order resample, seed, unit',
    seed: config.bootstrap.seed,
    percentile_indices: [Math.floor(0.025 * config.bootstrap.samples), Math.floor(0.975 * config.bootstrap.samples)],
    source_unit_mae_95_percent: Object.fromEntries(arms.map((arm) => [arm, interval(samples[arm])])),
    mixed_minus_raw_polynomial_mae_95_percent: interval(differences),
    mixed_over_zero_mae_95_percent: interval(ratios),
    limitation: 'Conditional on the three frozen fitted models; no population or training-process confidence guarantee.',
  };
}

/** Read-only audit. Audit invalidity and a valid negative benchmark screen differ. */
export async function auditAttempt(attemptDir, config) {
  const errors = [];
  let errorCount = 0;
  const checks = {};
  const maxima = {};
  const files = [];
  const fail = (message) => { errorCount++; if (errors.length < 80) errors.push(message); };
  const check = (condition, message, category = 'record_checks') => {
    checks[category] = (checks[category] ?? 0) + 1;
    if (!condition) fail(message);
  };
  const close = (actual, expected, label, category) => {
    const difference = Math.abs(actual - expected);
    maxima[category] = Math.max(maxima[category] ?? 0, difference);
    check(Number.isFinite(difference) && difference <= config.thresholds.numeric,
      `${label}: ${actual} differs from independent ${expected}`, category);
  };
  const read = async (filename) => {
    const bytes = await readFile(filename);
    files.push({ path: path.relative(attemptDir, filename).split(path.sep).join('/'), sha256: sha256(bytes), bytes: bytes.length });
    return bytes;
  };
  const report = {
    valid: false, supported: false,
    audit: { error_count: 0, errors, checks, max_abs: maxima, files },
    coverage: { development: {}, assessment: {} },
    metrics: { per_seed: {}, pooled: {} },
    uncertainty: null,
    limits: [
      'Synthetic static controlled-model-response benchmark only; no causal or real-system counterfactual claim.',
      'Error against analytic F includes frozen response-model fit error; it is distinct from contrast arithmetic error.',
      'Mixed and raw-polynomial features have equivalent expressive power; their comparison establishes no superiority.',
      'Three fixed seeds are all retained; conditional bootstrap does not characterize arbitrary training-seed variability.',
      'Library provenance claims are caller-supplied; this auditor verifies the retained benchmark artifacts and numeric records.',
    ],
  };
  const assessmentErrors = [];
  const allCoverage = { development: { requested: 0, success: 0, abstained: 0, failed: 0, unresolved: 0,
    in_support_requested: 0, in_support_success: 0, stress_requested: 0, stress_abstained: 0 },
  assessment: { requested: 0, success: 0, abstained: 0, failed: 0, unresolved: 0,
    in_support_requested: 0, in_support_success: 0, stress_requested: 0, stress_abstained: 0 } };
  try {
    for (const [key, value] of Object.entries(FIXED).filter(([key]) => !['seeds', 'counts', 'bootstrap'].includes(key))) {
      check(sameBinding(config[key], value), `Configuration ${key} differs from frozen interface`, 'configuration');
    }
    check(Array.isArray(config.seeds) && config.seeds.length > 0 && new Set(config.seeds).size === config.seeds.length
      && config.seeds.every((seed) => Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff), 'Invalid seed roster', 'configuration');
    check(SPLITS.every((split) => Number.isSafeInteger(config.counts?.[split]) && config.counts[split] > 0)
      && SPLITS.reduce((sum, split) => sum + config.counts[split], 0) <= 8193,
    'Invalid split counts', 'configuration');
    check(Number.isSafeInteger(config.bootstrap?.samples) && config.bootstrap.samples > 0 && config.bootstrap.seed === 9001,
      'Invalid bootstrap binding', 'configuration');
    report.scientific_configuration_matches_frozen_protocol = Object.entries(FIXED)
      .every(([key, value]) => sameBinding(config[key], value));
    if (errorCount) throw new Error('Audit cannot continue with a changed configuration');
    for (const seed of config.seeds) {
      const seedDir = path.join(attemptDir, `seed-${seed}`);
      const expected = regenerate(seed, config);
      const sources = {};
      const usedIds = new Set();
      const usedB = new Set();
      const sourceDigests = {};
      for (const split of SPLITS) {
        const bytes = await read(path.join(seedDir, SOURCE_FILES[split]));
        sourceDigests[split] = sha256(bytes);
        sources[split] = parseTable(bytes, INPUT_HEADER, `${seed}/${split}`).map((row, index) => ({
          unit: row.unit,
          a_num: integer(row.a_num, `${seed}/${split}/${index}/a`),
          b_num: integer(row.b_num, `${seed}/${split}/${index}/b`),
          noise_sign: integer(row.noise_sign, `${seed}/${split}/${index}/noise`),
        }));
        check(sources[split].length === config.counts[split], `${seed}/${split}: wrong row count`, 'source_rows');
        for (let i = 0; i < sources[split].length; i++) {
          const row = sources[split][i];
          check(JSON.stringify(row) === JSON.stringify(expected[split][i]), `${seed}/${split}/${i}: differs from independent generator`, 'generator_rows');
          check(!usedIds.has(row.unit), `${seed}: duplicate source ID ${row.unit}`, 'source_uniqueness');
          check(!usedB.has(row.b_num), `${seed}: repeated b across source splits`, 'source_uniqueness');
          check(row.a_num >= -3072 && row.a_num <= 3072 && row.b_num >= -4096 && row.b_num <= 4096
            && Math.abs(row.noise_sign) === 1, `${seed}/${row.unit}: invalid source bounds`, 'source_bounds');
          usedIds.add(row.unit); usedB.add(row.b_num);
        }
      }
      const modelBytes = await read(path.join(seedDir, 'model.tsv'));
      const metadataBytes = await read(path.join(seedDir, 'model.tsv.metadata.tsv'));
      const metadataRows = parseTable(metadataBytes, ['key', 'value'], `${seed}/model metadata`);
      const metadata = new Map(metadataRows.map((row) => [row.key, row.value]));
      check(metadata.size === metadataRows.length && metadataRows.length === 7, `${seed}: unexpected/duplicate model metadata`, 'model_metadata');
      const expectedMetadata = {
        study: 'OR06-SYNTH-RESP-001', version: '1.0.0', seed: String(seed),
        training_rows: String(config.counts.training), training_sha256: sourceDigests.training,
        model_sha256: sha256(modelBytes), fit_method: 'ols_normal_equations_partial_pivot_1e-12_train_population_sd',
      };
      for (const [key, value] of Object.entries(expectedMetadata)) {
        check(metadata.get(key) === value, `${seed}: incorrect model metadata ${key}`, 'model_metadata');
      }
      const modelRows = parseTable(modelBytes, MODEL_HEADER, `${seed}/model`);
      const expectedModelRows = config.arms.reduce((n, arm) => n + FEATURES[arm] + 1, 0);
      check(modelRows.length === expectedModelRows, `${seed}: wrong model row count`, 'model_rows');
      const models = {};
      let position = 0;
      for (const arm of config.arms) {
        const expectedFit = independentlyFit(sources.training, arm, config);
        const model = { means: [], scales: [], coefficients: [], intercept: 0 };
        for (let j = -1; j < FEATURES[arm]; j++, position++) {
          const row = modelRows[position];
          if (!row) throw new Error(`${seed}/${arm}: missing model row`);
          check(row.arm === arm && row.feature === (j < 0 ? 'intercept' : String(j)), `${seed}/${arm}: wrong model row order`, 'model_rows');
          const m = numeric(row.mean, 'model mean');
          const scale = numeric(row.scale, 'model scale');
          const coefficient = numeric(row.coefficient, 'model coefficient');
          check(scale > 0, `${seed}/${arm}: nonpositive model scale`, 'model_fit');
          close(m, j < 0 ? 0 : expectedFit.means[j], `${seed}/${arm}/${j}/mean`, 'train_only_scaler');
          close(scale, j < 0 ? 1 : expectedFit.scales[j], `${seed}/${arm}/${j}/scale`, 'train_only_scaler');
          close(coefficient, j < 0 ? expectedFit.intercept : expectedFit.coefficients[j], `${seed}/${arm}/${j}/coefficient`, 'train_only_fit');
          if (j < 0) model.intercept = coefficient;
          else { model.means.push(m); model.scales.push(scale); model.coefficients.push(coefficient); }
        }
        const coefficientVector = [model.intercept, ...model.coefficients];
        for (let j = 0; j < coefficientVector.length; j++) {
          close(expectedFit.gram[j].reduce((sum, x, k) => sum + x * coefficientVector[k], 0),
            expectedFit.right[j], `${seed}/${arm}/normal-equation/${j}`, 'normal_equation_residual');
        }
        models[arm] = model;
      }
      report.metrics.per_seed[seed] = {};
      for (const split of ['development', 'assessment']) {
        const outDir = path.join(seedDir, split);
        const pairs = parseTable(await read(path.join(outDir, 'pairs.tsv')), PAIR_HEADER, `${seed}/${split}/pairs`);
        const expectedCount = config.counts[split] * config.arms.length * config.changes.length;
        check(pairs.length === expectedCount, `${seed}/${split}: wrong pair roster count`, 'pair_rows');
        const coverage = { requested: pairs.length, success: 0, abstained: 0, failed: 0, unresolved: 0,
          in_support_requested: 0, in_support_success: 0, stress_requested: 0, stress_abstained: 0 };
        const unitErrors = Object.fromEntries([...config.arms, 'zero'].map((arm) => [arm, []]));
        let noChangeMaximum = 0;
        let arithmeticMaximum = 0;
        let pairPosition = 0;
        for (const arm of config.arms) {
          const model = models[arm];
          for (const source of sources[split]) {
            const errorsForUnit = [];
            const zeroForUnit = [];
            for (const change of config.changes) {
              const row = pairs[pairPosition++];
              if (!row) throw new Error(`${seed}/${split}: truncated pair roster`);
              const a = source.a_num / config.denominator;
              const b = source.b_num / config.denominator;
              const h = change / config.denominator;
              const altANum = source.a_num + change;
              const supported = Math.abs(altANum) <= config.denominator;
              check(row.arm === arm && row.unit === source.unit && integer(row.change_num, 'change') === change
                && integer(row.a_num, 'pair a') === source.a_num && integer(row.b_num, 'pair b') === source.b_num
                && integer(row.alt_a_num, 'alternative a') === altANum,
              `${seed}/${split}/${pairPosition}: incorrect source/arm/change/held-fixed binding`, 'pair_identity');
              if (Object.hasOwn(coverage, row.state)) coverage[row.state]++;
              else fail(`${seed}/${split}/${pairPosition}: unknown result state ${row.state}`);
              if (supported) { coverage.in_support_requested++; if (row.state === 'success') coverage.in_support_success++; }
              else { coverage.stress_requested++; if (row.state === 'abstained') coverage.stress_abstained++; }
              check(row.state === (supported ? 'success' : 'abstained')
                && row.reason === (supported ? 'NONE' : 'OUTSIDE_DECLARED_SUPPORT'),
              `${seed}/${split}/${pairPosition}: unexpected support state/reason`, 'pair_states');
              const baseFeatures = features(arm, a, b);
              const altFeatures = features(arm, altANum / config.denominator, b);
              for (const [name, rawExpected] of [['z_base', baseFeatures], ['z_alt', altFeatures]]) {
                const observed = row[name].split(',').map((value) => numeric(value, name));
                check(observed.length === rawExpected.length, `${seed}/${split}/${pairPosition}: wrong feature dimension`, 'raw_features');
                for (let j = 0; j < rawExpected.length; j++) close(observed[j], rawExpected[j], `${seed}/${split}/${pairPosition}/${name}/${j}`, 'raw_features');
              }
              const expectedBase = predict(model, baseFeatures);
              const expectedAlt = predict(model, altFeatures);
              const recordedBase = numeric(row.recorded_base, 'recorded baseline');
              const recordedAlt = numeric(row.recorded_alt, 'recorded alternative');
              close(recordedBase, expectedBase, `${seed}/${split}/${pairPosition}/baseline response`, 'saved_response');
              close(recordedAlt, expectedAlt, `${seed}/${split}/${pairPosition}/alternative response`, 'saved_response');
              if (supported) {
                const delta = numeric(row.delta, 'contrast');
                close(delta, expectedAlt - expectedBase, `${seed}/${split}/${pairPosition}/model contrast`, 'model_contrast');
                arithmeticMaximum = Math.max(arithmeticMaximum, Math.abs(delta - (expectedAlt - expectedBase)));
                const oracle = referenceContrast(a, b, h);
                errorsForUnit.push(Math.abs(delta - oracle));
                zeroForUnit.push(Math.abs(oracle));
                if (change === 0) { noChangeMaximum = Math.max(noChangeMaximum, Math.abs(delta)); close(delta, 0, 'no-change contrast', 'no_change'); }
              } else check(row.delta === 'NA', `${seed}/${split}/${pairPosition}: refusal has numeric contrast`, 'refusal_payloads');
            }
            check(errorsForUnit.length === 5, `${seed}/${split}/${source.unit}/${arm}: missing supported alternative`, 'unit_denominators');
            unitErrors[arm].push(mean(errorsForUnit));
            if (arm === config.arms[0]) unitErrors.zero.push(mean(zeroForUnit));
          }
        }
        for (const key of Object.keys(coverage)) allCoverage[split][key] += coverage[key];
        check(coverage.requested === coverage.success + coverage.abstained + coverage.failed + coverage.unresolved,
          `${seed}/${split}: result accounting does not reconcile`, 'coverage');
        report.metrics.per_seed[seed][split] = summary(unitErrors, coverage,
          { no_change_maximum_absolute_contrast: noChangeMaximum, model_contrast_maximum_absolute_error: arithmeticMaximum });
        if (split === 'assessment') assessmentErrors.push(unitErrors);
        const artifacts = parseTable(await read(path.join(outDir, 'artifacts.tsv')), ['sha256', 'payload_hex'], `${seed}/${split}/artifacts`);
        const store = new Map();
        for (const artifact of artifacts) {
          check(/^[0-9a-f]{64}$/.test(artifact.sha256), `${seed}/${split}: malformed artifact hash`, 'artifact_hashes');
          check(/^(?:[0-9a-f]{2})+$/.test(artifact.payload_hex), `${seed}/${split}: malformed artifact payload`, 'artifact_hashes');
          const payload = Buffer.from(artifact.payload_hex, 'hex');
          check(sha256(payload) === artifact.sha256, `${seed}/${split}: payload hash mismatch`, 'artifact_hashes');
          check(!store.has(artifact.sha256) || store.get(artifact.sha256).equals(payload), `${seed}/${split}: conflicting artifact content`, 'artifact_hashes');
          store.set(artifact.sha256, payload);
        }
        const descriptors = parseTable(await read(path.join(outDir, 'descriptors.tsv')),
          ['id', 'sha256', 'payload_hex'], `${seed}/${split}/descriptors`);
        const identityPins = new Map();
        for (const descriptor of descriptors) {
          const payload = Buffer.from(descriptor.payload_hex, 'hex');
          check(/^(?:[0-9a-f]{2})+$/.test(descriptor.payload_hex) && sha256(payload) === descriptor.sha256
            && store.has(descriptor.sha256) && store.get(descriptor.sha256).equals(payload),
          `${seed}/${split}/${descriptor.id}: descriptor/store bytes disagree`, 'descriptors');
          check(!identityPins.has(descriptor.id), `${seed}/${split}: duplicate descriptor ID`, 'descriptors');
          identityPins.set(descriptor.id, descriptor.sha256);
        }
        const roster = parseTable(await read(path.join(outDir, 'roster.tsv')),
          ['arm', 'pair_id', 'unit', 'change_num'], `${seed}/${split}/roster`);
        check(roster.length === pairs.length, `${seed}/${split}: saved roster differs in length`, 'saved_roster');
        const study = 'OR06-SYNTH-RESP-001';
        for (let i = 0; i < pairs.length; i++) {
          const pair = pairs[i]; const row = roster[i];
          check(row && row.arm === pair.arm && row.unit === pair.unit && row.change_num === pair.change_num
            && row.pair_id === `${study}/${seed}/${pair.arm}/${pair.unit}/change/${pair.change_num}`,
          `${seed}/${split}/${i}: saved roster differs from comparison records`, 'saved_roster');
        }
        const savedSummary = parseTable(await read(path.join(outDir, 'summary.tsv')),
          ['arm', 'requested', 'succeeded', 'abstained', 'failed', 'unresolved'], `${seed}/${split}/summary`);
        check(savedSummary.length === config.arms.length, `${seed}/${split}: wrong saved summary count`, 'saved_summary');
        for (let i = 0; i < config.arms.length; i++) {
          const arm = config.arms[i]; const row = savedSummary[i];
          if (!row) throw new Error('Missing saved summary row');
          const armPairs = pairs.filter((pair) => pair.arm === arm);
          check(row.arm === arm && integer(row.requested, 'requested') === armPairs.length
            && integer(row.succeeded, 'succeeded') === armPairs.filter((pair) => pair.state === 'success').length
            && integer(row.abstained, 'abstained') === armPairs.filter((pair) => pair.state === 'abstained').length
            && integer(row.failed, 'failed') === armPairs.filter((pair) => pair.state === 'failed').length
            && integer(row.unresolved, 'unresolved') === armPairs.filter((pair) => pair.state === 'unresolved').length,
          `${seed}/${split}/${arm}: saved summary is inconsistent`, 'saved_summary');
        }
        const recordMap = new Map();
        const recordKey = (arm, pair, scope, field) => `${arm}\t${pair}\t${scope}\t${field}`;
        for (const row of requestFields(await read(path.join(outDir, 'request_records.tsv')), `${seed}/${split}/request`)) {
          const key = recordKey(row.arm, row.pair_id, row.scope, row.field);
          check(config.arms.includes(row.arm) && !recordMap.has(key), `${seed}/${split}: duplicate/invalid serialized request field`, 'request_fields');
          recordMap.set(key, hexText(row.value_hex, `${seed}/${split}/${key}`));
        }
        const get = (arm, pair, scope, field) => {
          const key = recordKey(arm, pair, scope, field);
          if (!recordMap.has(key)) throw new Error(`Missing request field ${seed}/${split}/${key}`);
          return recordMap.get(key);
        };
        const equals = (arm, pair, scope, field, expectedValue) => {
          check(get(arm, pair, scope, field) === String(expectedValue),
            `${seed}/${split}/${arm}/${pair}/${scope}/${field}: incorrect serialized binding`, 'request_semantics');
        };
        const payloadOf = (arm, pair, scope, name) => {
          const digest = get(arm, pair, scope, `${name}.sha256`);
          if (!store.has(digest)) throw new Error(`Unresolved request artifact ${digest}`);
          return store.get(digest);
        };
        const kvPayload = (bytes, label) => {
          const text = bytes.toString('utf8');
          if (!Buffer.from(text, 'utf8').equals(bytes) || !text.endsWith('\n')) throw new Error(`${label}: invalid payload encoding`);
          const rows = text.slice(0, -1).split('\n').map((line) => line.split('\t'));
          if (rows.some((row) => row.length !== 2 || !row[0]) || new Set(rows.map((row) => row[0])).size !== rows.length) {
            throw new Error(`${label}: malformed payload fields`);
          }
          return Object.fromEntries(rows);
        };
        for (const [key] of recordMap) {
          if (!key.endsWith('.sha256')) continue;
          const [arm, pair_id, scope, field] = key.split('\t');
          const row = { arm, pair_id, scope, field };
          const name = row.field.slice(0, -7);
          const digest = get(row.arm, row.pair_id, row.scope, row.field);
          const id = get(row.arm, row.pair_id, row.scope, `${name}.id`);
          check(/^[0-9a-f]{64}$/.test(digest) && store.has(digest), `${seed}/${split}/${id}: unresolved ArtifactRef`, 'request_references');
          check(id.length > 0 && (!identityPins.has(id) || identityPins.get(id) === digest),
            `${seed}/${split}/${id}: conflicting identity pin`, 'request_references');
          identityPins.set(id, digest);
        }
        const requiredRoles = ['provider', 'preprocessing', 'fitted_transform', 'response_model', 'support_policy', 'constructor'];
        for (const arm of config.arms) {
          const prefix = `${study}/${seed}/${arm}`;
          const common = { request_id: `${prefix}/${split}/request`, operation_run_id: `${prefix}/${split}/run`,
            contract_id: 'OR06-PCR', contract_revision: '0.1.0', rer_id: 'OR06-PCR-RER', rer_revision: '0.1.0',
            claim_level: 'controlled_model_response', population_id: `${study}/${seed}/${split}/sha256/${sourceDigests[split]}`,
            sampling_rule: 'Every source unit, all six predeclared changes, no exclusions',
            response_horizon: 'static_synthetic_no_temporal_horizon', permitted_interpretation: 'model_input_response',
            permissions_ref: `${study}/synthetic_generated_inputs_no_external_data`,
            temporal_mode: 'non_temporal', as_of: 'NA', selected_output_position: '0', control_tolerance: '0' };
          for (const [field, value] of Object.entries(common)) equals(arm, '-', 'request', field, value);
          for (const field of ['purpose', 'resource_budget', 'evidence_destination', 'non_temporal_justification']) {
            check(get(arm, '-', 'request', field).length > 0, `${seed}/${split}/${arm}: missing ${field}`, 'request_semantics');
          }
          const sourceSHA = get(arm, '-', 'request', 'code_sha256');
          const configSHA = get(arm, '-', 'request', 'configuration_sha256');
          check(/^[0-9a-f]{64}$/.test(sourceSHA) && /^[0-9a-f]{64}$/.test(configSHA), 'Malformed study source/config binding', 'request_semantics');
          if (!report.bindings) report.bindings = { source_sha256: sourceSHA, configuration_sha256: configSHA };
          check(report.bindings.source_sha256 === sourceSHA && report.bindings.configuration_sha256 === configSHA,
            'Study source/config binding changed between requests', 'request_semantics');
          equals(arm, '-', 'pipeline', 'configuration_sha256', configSHA);
          equals(arm, '-', 'pipeline', 'stochastic_policy', 'deterministic');
          equals(arm, '-', 'pipeline', 'provider_revision_known', 'true');
          equals(arm, '-', 'pipeline', 'response_model.sha256', sha256(modelBytes));
          equals(arm, '-', 'expected_space', 'fitted_transform.sha256', sha256(modelBytes));
          const expectedSpace = { output_selector: 'features', native_shape: String(FEATURES[arm]), layout: 'dense',
            normalization: 'train_population_zscore_in_response_head', geometry: 'euclidean', input_role: 'response_input' };
          for (const [field, value] of Object.entries(expectedSpace)) equals(arm, '-', 'expected_space', field, value);
          for (const role of requiredRoles) {
            const scope = `dependency.${role}`;
            equals(arm, '-', scope, 'role', role);
            equals(arm, '-', scope, 'selection_population', 'none_no_selection');
            equals(arm, '-', scope, 'pretraining_exposure', 'excluded_by_design');
            equals(arm, '-', scope, 'available_at', 'NA');
            const fitted = ['preprocessing', 'fitted_transform', 'response_model'].includes(role);
            equals(arm, '-', scope, 'fit_population', fitted
              ? `${study}/${seed}/training/sha256/${sourceDigests.training}` : 'none_fixed_by_design');
            if (['provider', 'preprocessing', 'fitted_transform'].includes(role)) {
              equals(arm, '-', scope, 'artifact.sha256', get(arm, '-', 'expected_space', `${role}.sha256`));
            } else if (role === 'response_model') equals(arm, '-', scope, 'artifact.sha256', sha256(modelBytes));
          }
          const provider = kvPayload(payloadOf(arm, '-', 'expected_space', 'provider'), 'provider descriptor');
          const featureDefinitions = { mixed: 'a+b,a-b,ab+a^2,ab-a^2', raw_polynomial: 'a,b,ab,a^2', raw_linear: 'a,b', lossy: 'b,b^2,b^3,b^4' };
          check(provider.arm === arm && provider.features === featureDefinitions[arm] && provider.provider === 'fixed_exact_arithmetic_v1'
            && provider.source_sha256 === sourceSHA && provider.configuration_sha256 === configSHA && provider.seed === String(seed),
          `${seed}/${split}/${arm}: invalid provider descriptor`, 'descriptor_semantics');
          const preprocessing = kvPayload(payloadOf(arm, '-', 'expected_space', 'preprocessing'), 'preprocessing descriptor');
          check(preprocessing.method === 'training_population_mean_and_sd' && preprocessing.model_sha256 === sha256(modelBytes)
            && preprocessing.fit_metadata_sha256 === sha256(metadataBytes),
          `${seed}/${split}/${arm}: invalid preprocessing descriptor`, 'descriptor_semantics');
          const spacePayload = kvPayload(payloadOf(arm, '-', 'expected_space', 'artifact'), 'space descriptor');
          for (const [field, value] of Object.entries(expectedSpace)) {
            check(spacePayload[field] === value, `${seed}/${split}/${arm}: space descriptor ${field} differs`, 'descriptor_semantics');
          }
          for (const role of ['provider', 'preprocessing', 'fitted_transform']) {
            check(spacePayload[`${role}_sha256`] === get(arm, '-', 'expected_space', `${role}.sha256`),
              `${seed}/${split}/${arm}: space dependency pin differs`, 'descriptor_semantics');
          }
          const pipelinePayload = kvPayload(payloadOf(arm, '-', 'pipeline', 'artifact'), 'pipeline descriptor');
          for (const role of requiredRoles) {
            check(pipelinePayload[`${role}_sha256`] === get(arm, '-', `dependency.${role}`, 'artifact.sha256'),
              `${seed}/${split}/${arm}: pipeline dependency pin differs`, 'descriptor_semantics');
          }
          check(pipelinePayload.stochastic_policy === 'deterministic' && pipelinePayload.provider_revision_known === 'true',
            `${seed}/${split}/${arm}: pipeline determinism descriptor differs`, 'descriptor_semantics');
          const constructorPayload = kvPayload(payloadOf(arm, '-', 'dependency.constructor', 'artifact'), 'constructor descriptor');
          check(constructorPayload.method === 'add_change_num_to_a_num_hold_b_num_exact'
            && constructorPayload.changes === config.changes.join(','), `${seed}/${split}/${arm}: alternative policy differs`, 'descriptor_semantics');
          const supportPayload = kvPayload(payloadOf(arm, '-', 'dependency.support_policy', 'artifact'), 'support descriptor');
          check(supportPayload.a_min === '-1' && supportPayload.a_max === '1' && supportPayload.b_min === '-1' && supportPayload.b_max === '1',
            `${seed}/${split}/${arm}: support bounds differ`, 'descriptor_semantics');
          for (let j = 0; j < 2; j++) {
            const scope = `attribute_rule.${j}`;
            for (const [field, value] of Object.entries({ name: j ? 'b' : 'a', units: 'dimensionless', role: j ? 'held_fixed' : 'changed', minimum: '-1', maximum: '1' })) {
              equals(arm, '-', scope, field, value);
            }
          }
          for (const [field, value] of Object.entries({ id: 'synthetic_response', definition_revision: 'synthetic_response_units_v1', units: 'synthetic_unit' })) {
            equals(arm, '-', 'response_axis.0', field, value);
          }
          for (const row of pairs.filter((row) => row.arm === arm)) {
            const pairId = `${prefix}/${row.unit}/change/${row.change_num}`;
            const source = sources[split].find((source) => source.unit === row.unit);
            const changedA = integer(row.alt_a_num, 'serialized alternate a');
            const change = integer(row.change_num, 'serialized change');
            for (const [field, value] of Object.entries({ id: pairId, unit_id: row.unit, entity_id: `synthetic_entity/${row.unit}`,
              group_id: `synthetic_seed/${seed}`, interpretation: 'model_input_response',
              change_definition: `a_num += ${change}; b_num held exactly fixed; denominator=4096` })) equals(arm, pairId, 'pair', field, value);
            equals(arm, pairId, 'pair', 'constructor.sha256', get(arm, '-', 'dependency.constructor', 'artifact.sha256'));
            for (const [scope, state] of [['alternative_validity', 'passed'], ['interpretability', 'passed'],
              ['support', Math.abs(changedA) <= config.denominator ? 'passed' : 'rejected']]) equals(arm, pairId, scope, 'state', state);
            const control = kvPayload(payloadOf(arm, pairId, 'pair', 'control_evidence'), 'control evidence');
            check(control.policy_sha256 === get(arm, pairId, 'pair', 'constructor.sha256')
              && control.base_b_num === String(source.b_num) && control.alt_b_num === String(source.b_num) && control.output_position === '0',
            `${pairId}: incorrect control evidence`, 'control_evidence');
            const validity = kvPayload(payloadOf(arm, pairId, 'alternative_validity', 'evidence'), 'validity evidence');
            check(validity.constructor_sha256 === get(arm, pairId, 'pair', 'constructor.sha256') && validity.base_a_num === String(source.a_num)
              && validity.change_num === String(change) && validity.alt_a_num === String(changedA) && validity.result === 'passed',
            `${pairId}: incorrect validity evidence`, 'control_evidence');
            const supportEvidence = kvPayload(payloadOf(arm, pairId, 'support', 'evidence'), 'support evidence');
            check(supportEvidence.support_policy_sha256 === get(arm, '-', 'dependency.support_policy', 'artifact.sha256')
              && supportEvidence.base_a_num === String(source.a_num) && supportEvidence.alt_a_num === String(changedA)
              && supportEvidence.b_num === String(source.b_num) && supportEvidence.result === (Math.abs(changedA) <= config.denominator ? 'passed' : 'rejected'),
            `${pairId}: incorrect support evidence`, 'control_evidence');
            const interpretationEvidence = kvPayload(payloadOf(arm, pairId, 'interpretability', 'evidence'), 'interpretability evidence');
            check(interpretationEvidence.interpretation === 'model_input_response' && interpretationEvidence.claim === 'controlled_model_response'
              && interpretationEvidence.pipeline_sha256 === get(arm, '-', 'pipeline', 'artifact.sha256') && interpretationEvidence.result === 'passed',
            `${pairId}: incorrect interpretation evidence`, 'control_evidence');
            for (const scope of ['baseline', 'alternative']) {
              const isBase = scope === 'baseline';
              const aNum = isBase ? source.a_num : changedA;
              const branchPrefix = isBase ? `${prefix}/${row.unit}/baseline` : `${pairId}/alternative`;
              for (const name of ['input', 'representation', 'response']) equals(arm, pairId, scope, `${name}.id`, `${branchPrefix}/${name}`);
              equals(arm, pairId, scope, 'response_execution_id', `${branchPrefix}/execution`);
              equals(arm, pairId, scope, 'origin', isBase ? 'observed' : 'edited_input');
              equals(arm, pairId, scope, 'representation_kind', isBase ? 'observation_embedding' : 'transformed_candidate');
              equals(arm, pairId, scope, 'parent_input_id', isBase ? '' : get(arm, pairId, 'baseline', 'input.id'));
              equals(arm, pairId, scope, 'parent_representation_id', isBase ? '' : get(arm, pairId, 'baseline', 'representation.id'));
              for (const [field, value] of Object.entries({ output_position: '0', output_valid: 'true', upstream_succeeded: 'true',
                'timing.support_start': 'NA', 'timing.support_end': 'NA', 'timing.available_at': 'NA',
                'timing.vintage': `${study}/static_generated_v1` })) equals(arm, pairId, scope, field, value);
              equals(arm, pairId, scope, 'pipeline.sha256', get(arm, '-', 'pipeline', 'artifact.sha256'));
              for (const field of [...Object.keys(expectedSpace), 'artifact.id', 'artifact.sha256', 'provider.id', 'provider.sha256',
                'preprocessing.id', 'preprocessing.sha256', 'fitted_transform.id', 'fitted_transform.sha256']) {
                equals(arm, pairId, `${scope}.space`, field, get(arm, '-', 'expected_space', field));
              }
              const inputBytes = Buffer.from(`a_num\t${aNum}\nb_num\t${source.b_num}\n`);
              check(payloadOf(arm, pairId, scope, 'input').equals(inputBytes), `${pairId}/${scope}: input bytes mismatch`, 'branch_payloads');
              const sourceBytes = Buffer.from(`${source.unit}\t${source.a_num}\t${source.b_num}\t${source.noise_sign}\n`);
              check(payloadOf(arm, pairId, scope, 'source_observation').equals(sourceBytes), `${pairId}/${scope}: source bytes mismatch`, 'branch_payloads');
              equals(arm, pairId, scope, 'source_observation.id', `${study}/${seed}/source/${row.unit}`);
              const representation = kvPayload(payloadOf(arm, pairId, scope, 'representation'), 'representation payload');
              check(representation.arm === arm, `${pairId}/${scope}: representation arm mismatch`, 'branch_payloads');
              const raw = features(arm, aNum / config.denominator, source.b_num / config.denominator);
              const savedFeatures = representation.z.split(',').map((value) => numeric(value, 'retained feature'));
              check(savedFeatures.length === raw.length, `${pairId}/${scope}: retained feature shape mismatch`, 'branch_payloads');
              raw.forEach((value, j) => close(savedFeatures[j], value, `${pairId}/${scope}/retained feature/${j}`, 'branch_payloads'));
              const response = kvPayload(payloadOf(arm, pairId, scope, 'response'), 'response payload');
              check(response.arm === arm && response.model_sha256 === sha256(modelBytes), `${pairId}/${scope}: response model mismatch`, 'branch_payloads');
              const expectedResponse = numeric(isBase ? row.recorded_base : row.recorded_alt, 'comparison response');
              close(numeric(response.value, 'retained response'), expectedResponse, `${pairId}/${scope}/retained response`, 'branch_payloads');
              close(numeric(get(arm, pairId, scope, 'response_values'), 'request response'), expectedResponse, `${pairId}/${scope}/request response`, 'branch_payloads');
              check(payloadOf(arm, pairId, scope, 'mask').equals(Buffer.from('output_position\t0\nvalid\ttrue\n')),
                `${pairId}/${scope}: mask bytes mismatch`, 'branch_payloads');
              for (let j = 0; j < 2; j++) {
                equals(arm, pairId, scope, `attributes.${j}.name`, j ? 'b' : 'a');
                equals(arm, pairId, scope, `attributes.${j}.units`, 'dimensionless');
                close(numeric(get(arm, pairId, scope, `attributes.${j}.value`), 'attribute value'),
                  (j ? source.b_num : aNum) / config.denominator, `${pairId}/${scope}/attribute/${j}`, 'branch_payloads');
              }
              for (const field of ['id', 'definition_revision', 'units']) {
                equals(arm, pairId, scope, `axes.0.${field}`, get(arm, '-', 'response_axis.0', field));
              }
            }
          }
        }
      }
    }
    report.coverage = allCoverage;
    const arms = [...config.arms, 'zero'];
    for (const split of ['development', 'assessment']) {
      const scores = Object.fromEntries(arms.map((arm) => [arm, mean(config.seeds.map((seed) => report.metrics.per_seed[seed][split].source_unit_mae[arm]))]));
      report.metrics.pooled[split] = { source_units: config.seeds.length * config.counts[split],
        aggregation: 'equal source-unit weighting; equal split sizes make this the seed macro-average',
        source_unit_mae: scores };
    }
    report.uncertainty = bootstrap(assessmentErrors, config);
    const pooled = report.metrics.pooled.assessment.source_unit_mae;
    const criteria = {
      mixed_mae_each_seed: config.seeds.every((seed) => report.metrics.per_seed[seed].assessment.source_unit_mae.mixed <= config.thresholds.mixed_mae),
      mixed_raw_polynomial_equivalence_each_seed: config.seeds.every((seed) => Math.abs(report.metrics.per_seed[seed].assessment.source_unit_mae.mixed
        - report.metrics.per_seed[seed].assessment.source_unit_mae.raw_polynomial) <= config.thresholds.equivalence_mae),
      pooled_mixed_over_zero: pooled.mixed <= config.thresholds.zero_ratio * pooled.zero,
      in_support_coverage_complete: allCoverage.assessment.in_support_success === allCoverage.assessment.in_support_requested,
      stress_abstention_complete: allCoverage.assessment.stress_abstained === allCoverage.assessment.stress_requested,
      independent_numerical_audit: errorCount === 0,
    };
    report.screen = { supported: Object.values(criteria).every(Boolean), criteria,
      interpretation: 'Valid negative/inconclusive screen remains a completed evaluation; invalid audit does not.' };
  } catch (error) { fail(error.message); }
  report.audit.error_count = errorCount;
  report.valid = errorCount === 0;
  report.supported = report.valid && Boolean(report.screen?.supported);
  if (!report.screen) report.screen = { supported: false, criteria: { independent_numerical_audit: false } };
  return report;
}
