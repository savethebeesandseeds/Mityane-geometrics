// Engineering saved-vector requests. No encoder, dataset, or provider call.
import {createHash} from 'node:crypto';
import {jsonBytes} from '../../../shared/artifacts/node/v1/files.mjs';

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const identity = text => digest(Buffer.from(text));

function artifact(id, body) {
  const bytes = jsonBytes(body);
  return {id, sha256: digest(bytes), bytes, body};
}

function member(sourceId, values, timing = {}) {
  const representation_kind = 'observation_embedding';
  const saved = artifact(`parent-${sourceId}`, {source_id: sourceId, representation_kind, values});
  return {
    record: {
      source_id: sourceId, parent: {id: saved.id, sha256: saved.sha256}, representation_kind, values,
      available_at: timing.available_at ?? null,
      window_start: timing.window_start ?? null,
      window_end: timing.window_end ?? null,
    },
    artifact: saved,
  };
}

export function engineeringRequest(spec) {
  const provider = spec.provider ?? 'toy-a';
  const selected = provider === 'toy-a'
    ? {provider_id: 'synthetic-saved-vectors', dimensions: 2, metric: 'l2', normalization: 'none',
      query: [0, 0], corpus: [{source_id: 'near', values: [1, 0]}, {source_id: 'far', values: [4, 0]}]}
    : provider === 'toy-b'
      ? {provider_id: 'synthetic-saved-vectors', dimensions: 4, metric: 'cosine', normalization: 'l2',
        query: [1, 0, 0, 0], corpus: [{source_id: 'same', values: [2, 0, 0, 0]}, {source_id: 'other', values: [0, 3, 0, 0]}]}
      : null;
  if (!selected) throw new Error(`Unknown engineering provider ${provider}; use toy-a or toy-b.`);
  const chosen = {...selected, ...spec, provider_id: spec.provider_id ?? selected.provider_id};
  const profile = chosen.profile ?? 'local-saved-vectors';
  const spaceBody = {
    space_id: chosen.space_id ?? 'space-v1',
    snapshot_id: chosen.snapshot_id ?? 'snapshot-v1',
    provider_id: chosen.provider_id,
    revision_id: chosen.revision_id ?? 'engineering-revision-v1',
    dimensions: chosen.dimensions,
    normalization: chosen.normalization,
    metric: chosen.metric,
    task_role: chosen.task_role ?? 'saved-vector',
    representation_kind: 'observation_embedding',
  };
  const spaceArtifact = artifact('space-artifact', spaceBody);
  const providerArtifact = artifact('provider-artifact', {provider_id: spaceBody.provider_id});
  const revisionArtifact = artifact('revision-artifact', {revision_id: spaceBody.revision_id});
  const fittedArtifact = artifact('fitted-artifact', {available_at: chosen.fitted_available_at ?? null});
  const queryValues = chosen.query;
  const queryMember = member(chosen.query_source_id ?? 'query-source', queryValues, {
    available_at: chosen.query_available_at ?? null,
    window_start: chosen.query_window?.[0] ?? null,
    window_end: chosen.query_window?.[1] ?? null,
  });
  const query = {
    id: chosen.query_id ?? 'query-1', source_id: queryMember.record.source_id, parent: queryMember.record.parent,
    representation_kind: queryMember.record.representation_kind, values: queryValues,
    available_at: queryMember.record.available_at,
    window_start: queryMember.record.window_start,
    window_end: queryMember.record.window_end,
  };
  const built = (chosen.corpus ?? selected.corpus).map(row => member(row.source_id, row.values, row));
  const input = {
    request_id: chosen.request_id ?? 'request-engineering-1',
    operation_run_id: chosen.operation_run_id ?? 'run-engineering-1',
    contract_id: 'OR01-RET', contract_revision: '0.1.0',
    rer_id: 'OR01-RET-RER', rer_revision: '0.1.0',
    claim_level: chosen.claim_level ?? 'exploratory_description',
    profile, live_call: chosen.live_call ?? false,
    purpose: 'engineering-only exact search',
    code_sha256: identity('or01-ret-engineering-code'),
    configuration_sha256: identity('or01-ret-engineering-configuration'),
    evidence_destination: 'engineering',
    temporal_mode: chosen.temporal_mode ?? 'non_temporal',
    as_of: chosen.as_of ?? null,
    non_temporal_justification: chosen.non_temporal_justification ??
      'Saved vectors supplied without a time index.',
    space: {
      artifact: {id: spaceArtifact.id, sha256: spaceArtifact.sha256},
      provider: {id: providerArtifact.id, sha256: providerArtifact.sha256},
      revision: {id: revisionArtifact.id, sha256: revisionArtifact.sha256},
      fitted_transform: {id: fittedArtifact.id, sha256: fittedArtifact.sha256},
      ...spaceBody,
      fitted_available_at: chosen.fitted_available_at ?? null,
    },
    query, corpus: built.map(row => row.record),
    k: chosen.k ?? 2, exclude_self: chosen.exclude_self ?? true,
    excluded_source_ids: chosen.excluded_source_ids ?? [],
    overlap_policy: chosen.overlap_policy ?? 'none',
    relevance_rule: null,
  };
  return {
    input,
    artifacts: [spaceArtifact, providerArtifact, revisionArtifact, fittedArtifact, queryMember.artifact,
      ...built.map(row => row.artifact)],
  };
}
