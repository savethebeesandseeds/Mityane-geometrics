// Metadata CLI. Archived original source and historical seals stay intact.
import { main as capsuleMain } from '../node/v1/capsule_cli.mjs';
import { isMain } from '../node/v1/metadata.mjs';
import { createRelationCapsule } from '../node/v1/relation_capsule.mjs';

export const PROTOCOL = 'configurable-learned-relation-v1';
export const FAMILY = Object.freeze({ protocol: PROTOCOL, directory: 'configurable_learned_relation_v1',
  runName: 'rpb-v21-learned-relation', content: false,
  card: 'code/embeddings/evaluation/cards/configurable_learned_relation_v1.md',
  config: 'code/embeddings/encoders/raw_patch_bottleneck_mae/versions/v21/config/learned_ordered_relation.conf',
  binaries: Object.freeze(['embedding_configurable_learned_relation', 'embedding_configurable_learned_relation_readout',
    'code/tests/encoders/raw_patch_bottleneck_mae/learned_ordered_relation/learned_ordered_relation_test']) });

// options binds current repository/build paths, mapped historical capsules,
// captured Git revision, the complete source-list text, and a source-id runner.
// No default maps an old /embedding identity onto current source silently.
export const createCapsule = options => createRelationCapsule(options, FAMILY);
if(isMain(import.meta.url)) await capsuleMain('configurable_learned_relation_v1',createCapsule);
