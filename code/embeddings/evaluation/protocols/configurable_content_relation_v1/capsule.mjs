// Metadata CLI. Archived original source and historical seals stay intact.
import { main as capsuleMain } from '../node/v1/capsule_cli.mjs';
import { isMain } from '../node/v1/metadata.mjs';
import { createRelationCapsule } from '../node/v1/relation_capsule.mjs';

export const PROTOCOL = 'configurable-content-relation-v1';
export const FAMILY = Object.freeze({ protocol: PROTOCOL, directory: 'configurable_content_relation_v1',
  runName: 'rpb-v22-content-relation', content: true,
  card: 'code/embeddings/evaluation/cards/configurable_content_relation_v1.md',
  config: 'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v22/config/content_ordered_relation.conf',
  binaries: Object.freeze(['embedding_configurable_content_relation', 'embedding_configurable_content_relation_readout',
    'code/tests/encoders/raw_patch_bottleneck_mae/content_ordered_relation/content_ordered_relation_test']) });

export const createCapsule = options => createRelationCapsule(options, FAMILY);
if(isMain(import.meta.url)) await capsuleMain('configurable_content_relation_v1',createCapsule);
