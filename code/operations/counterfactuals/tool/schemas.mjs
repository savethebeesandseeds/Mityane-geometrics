// SPDX-License-Identifier: MIT
// OR06-PCR owns every native field and diagnostic. These schemas describe the
// recorded comparator, not observational or identified intervention profiles.
import {summarySchema} from '../../../shared/representation/tools/v1/schemas.mjs';

const object = properties => ({type: 'object', additionalProperties: false,
  required: Object.keys(properties), properties});
const text = {type: 'string'};
const numbers = {type: 'array', minItems: 1, maxItems: 10000, items: {type: 'number'}};
const nullable = schema => ({anyOf: [{type: 'null'}, schema]});
const count = {type: 'integer', minimum: 0, maximum: 10000};
const diagnosticSchema = object({
  code: {type: 'string', enum: ['UNSUPPORTED_CLAIM', 'INVALID_RECORD', 'INVALID_PAIRING',
    'INCOMPATIBLE_ROLE_OR_SPACE', 'STALE_OR_CHANGED_PIPELINE', 'UNVERIFIED_CONTROL',
    'TEMPORAL_VIOLATION', 'UPSTREAM_FAILURE', 'NONFINITE_RESPONSE_OR_CONTRAST',
    'INVALID_ALTERNATIVE', 'OUTSIDE_DECLARED_SUPPORT', 'UNSUPPORTED_INTERPRETATION', 'INTERRUPTED']},
  record_id: text, message: text,
});
const diagnostics = {type: 'array', items: diagnosticSchema};
const contrastSchema = object({baseline: numbers, alternative: numbers, delta: numbers});
const pairResultSchema = {
  ...object({pair_index: {type: 'integer', minimum: 0, maximum: 9999}, pair_id: text,
    state: {type: 'string', enum: ['success', 'abstained', 'failed', 'unresolved']},
    diagnostics, contrast: nullable(contrastSchema),
    available_at: nullable({type: 'integer', minimum: -Number.MAX_SAFE_INTEGER, maximum: Number.MAX_SAFE_INTEGER})}),
  allOf: [{if: {properties: {state: {const: 'success'}}},
    then: {properties: {contrast: contrastSchema}}, else: {properties: {contrast: {type: 'null'}}}}],
};
const nativeSummary = {...summarySchema,
  properties: {...summarySchema.properties, requested: count, succeeded: count,
    abstained: count, failed: count, unresolved: count}};

export const nativeResultSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:native-result:1',
  title: 'OR06-PCR recorded paired-response native result v1',
  description: 'Exact native.cpp JSON output. Full denominator, pair order, matching vector widths, '
    + 'signed arithmetic and request identity are additionally verified by the operation adapter. '
    + 'The separately retained request contains the input provenance. No causal claim follows.',
  ...object({admitted: {type: 'boolean'}, admission_diagnostics: diagnostics,
    pairs: {type: 'array', maxItems: 10000, items: pairResultSchema}, summary: nativeSummary}),
  allOf: [{if: {properties: {admitted: {const: false}}},
    then: {properties: {pairs: {maxItems: 0}, admission_diagnostics: {minItems: 1},
      summary: {properties: {succeeded: {const: 0}, abstained: {const: 0},
        failed: {const: 0}, unresolved: {const: 0}, success_rate: {type: 'null'}}}}},
    else: {properties: {pairs: {minItems: 1}, admission_diagnostics: {maxItems: 0},
      summary: {properties: {requested: {minimum: 1}, success_rate: {type: 'number'}}}}}}],
};

export const responseArtifactSchema = {
  $schema: 'https://json-schema.org/draft/2020-12/schema',
  $id: 'urn:mityane:operations:paired-response:response-artifact:1',
  title: 'OR06-PCR recorded numerical response artifact v1',
  description: 'The adapter verifies these axes and values equal the corresponding inline branch fields exactly.',
  ...object({axes: {type: 'array', maxItems: 10000,
    items: object({id: text, definition_revision: text, units: text})},
  values: {type: 'array', maxItems: 10000, items: {type: 'number'}}}),
};
