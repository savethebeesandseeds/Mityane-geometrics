// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Source inventory/copy/verification only. No model execution or data loading.
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {exportBundle, verifyBundle} from './export-cpb-family.mjs';

export const V19_PROFILE='code/embeddings/encoders/causal_patch_bottleneck_mae/versions/v19/migration/package.json';
export function exportV19({root,output}) {
  return exportBundle({root,output,profilePath:V19_PROFILE});
}
export function main(argv=process.argv.slice(2)) {
  const usage='Usage: export-cpb-v19.mjs --output NEWLEAF\n'+
      '   or: export-cpb-v19.mjs --verify BUNDLE --expected-manifest-sha256 SHA256';
  if(argv.length===1&&argv[0]==='--help'){process.stdout.write(`${usage}\n`);return;}
  const args=new Map();
  if(!argv.length||argv.length%2)throw new Error(usage);
  for(let i=0;i<argv.length;i+=2){
    if(!['--output','--verify','--expected-manifest-sha256'].includes(argv[i])||args.has(argv[i])||!argv[i+1])
      throw new Error(usage);
    args.set(argv[i],argv[i+1]);
  }
  let result;
  if(args.size===1&&args.has('--output')) {
    result=exportV19({root:path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../..'),output:args.get('--output')});
  } else if(args.size===2&&args.has('--verify')&&args.has('--expected-manifest-sha256')) {
    result=verifyBundle({bundle:args.get('--verify'),expectedManifestSha256:args.get('--expected-manifest-sha256')});
  } else throw new Error(usage);
  process.stdout.write(`${JSON.stringify(result,null,2)}\n`);
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  try{main();}catch(error){process.stderr.write(`[CPB-v19 export] ${error.message}\n`);process.exitCode=1;}
}
