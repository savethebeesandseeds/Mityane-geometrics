// OR01-RET native result. Exploratory saved-vector ranking, not a relevance verdict.
import {summarySchema} from '../../../shared/representation/tools/v1/schemas.mjs';

const object = properties => ({type: 'object', additionalProperties: false,
  required: Object.keys(properties), properties});
const text = {type: 'string', minLength: 1};
const count = {type: 'integer', minimum: 0, maximum: 4096};
const diagnosticSchema = object({
  code: {type: 'string', enum: ['UNSUPPORTED_CLAIM', 'LIVE_CALL_REFUSED', 'RELEVANCE_UNBOUND',
    'INVALID_RECORD', 'INCOMPATIBLE_ROLE_OR_SPACE', 'TEMPORAL_VIOLATION', 'STALE_FITTED_ARTIFACT',
    'DUPLICATE_SOURCE', 'NONFINITE_VALUE', 'DIMENSION_MISMATCH', 'SOURCE_WINDOW_OVERLAP']},
  record_id: text, message: text,
});
const hitSchema = object({
  source_id: text, parent_id: text, representation_kind: text,
  rank: {type: 'integer', minimum: 1, maximum: 4096},
  distance: {type: 'number'}, similarity: {type: 'number'},
});
const skipSchema = object({
  source_id: text, parent_id: text,
  reason: {type: 'string', enum: ['self_match', 'excluded_source', 'undefined_geometry',
    'temporal_violation', 'source_window_overlap']},
});

export const nativeResultSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:retrieval:native-result:1',
  title: 'OR01-RET exact-search native result v1',
  description: 'Engineering ranking of saved vectors. engineering_only is always true. '
    + 'Empty, abstained, and failed stay distinct. A rejected admission contains no hits and no skips.',
  ...object({
    admitted: {type: 'boolean'},
    engineering_only: {const: true},
    admission_diagnostics: {type: 'array', maxItems: 64, items: diagnosticSchema},
    operation_run_id: text, claim_level: text, profile: text,
    space_id: text, snapshot_id: text, metric: text, normalization: text,
    code_sha256: {type: 'string', pattern: '^[0-9a-f]{64}$'},
    configuration_sha256: {type: 'string', pattern: '^[0-9a-f]{64}$'},
    fitted_transform_id: text,
    outcome: {type: 'string', enum: ['ranked', 'empty', 'abstained', 'failed', 'rejected']},
    hits: {type: 'array', maxItems: 4096, items: hitSchema},
    skipped: {type: 'array', maxItems: 4096, items: skipSchema},
    eligible_count: count,
    summary: summarySchema,
  }),
  allOf: [{if: {properties: {admitted: {const: false}}},
    then: {properties: {outcome: {const: 'rejected'}, hits: {maxItems: 0}, skipped: {maxItems: 0},
      eligible_count: {const: 0}, admission_diagnostics: {minItems: 1}}}}],
};

export const vectorArtifactSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:retrieval:vector-artifact:1',
  title: 'OR01-RET saved vector artifact v1',
  description: 'The adapter requires these values to equal the inline query or corpus member.',
  ...object({
    source_id: text,
    representation_kind: text,
    values: {type: 'array', maxItems: 8192, items: {type: 'number'}},
  }),
};
