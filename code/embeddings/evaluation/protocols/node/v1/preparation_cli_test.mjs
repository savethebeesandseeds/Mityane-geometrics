// Artificial CLI integration only. Opaque tensor roles are never decoded.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {FAMILIES, SDK, inputNames} from './metadata.mjs';
import {JSON_ENCODING, sha256} from '../../../../../shared/artifacts/node/v1/files.mjs';

const repository='/mityane', prefix='code/embeddings/evaluation/protocols';
function fixture(t) {
  assert.equal(process.platform,'linux');assert.ok(fs.existsSync('/.dockerenv'));
  assert.equal(process.cwd(),repository);assert.ok(process.env.TMPDIR?.startsWith('/opt/mityane-geometrics/'));
  const root=fs.mkdtempSync(path.join(process.env.TMPDIR,'metadata-positive-cli-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  function write(name,value) {
    const target=path.join(root,name);fs.mkdirSync(path.dirname(target),{recursive:true});
    fs.writeFileSync(target,typeof value==='string'?value:JSON.stringify(value)+'\n');return target;
  }
  function copy(name) {
    const target=path.join(root,name);fs.mkdirSync(path.dirname(target),{recursive:true});
    fs.copyFileSync(path.join(repository,name),target);
  }
  return {root,write,copy};
}
function invoke(family,name,args) {
  const result=spawnSync(process.execPath,[`${repository}/${prefix}/${family}/${name}.mjs`,...args],
    {cwd:repository,encoding:'utf8',timeout:15000,maxBuffer:1024*1024});
  assert.equal(result.error,undefined);assert.equal(result.signal,null);return result;
}
function passed(result) {
  assert.equal(result.status,0,result.stderr);assert.equal(result.stderr,'');return JSON.parse(result.stdout);
}
const archivedResults='.archive/embeddings/pre-cleanup-2026-10-18/doc/results/';
for(const [family,policy] of Object.entries(FAMILIES))test(`${family} prepares an exclusive engineering freeze through its real CLI`,t=>{
  const f=fixture(t),familyPrefix=`${prefix}/${family}/`;
  for(const entry of fs.readdirSync(path.join(repository,familyPrefix)))if(entry.endsWith('.mjs'))f.copy(familyPrefix+entry);
  for(const name of [`${prefix}/node/v1/metadata.mjs`,'code/shared/artifacts/node/v1/files.mjs',
    'code/scripts/node.sh','code/scripts/node-runtime.sh'])f.copy(name);
  f.write('fixture.cpp','// inert artificial source\n');
  f.write(`code/embeddings/evaluation/cards/${family}.md`,'Artificial engineering card; no execution authorization.\n');
  if(policy.tasks)f.copy(archivedResults+'two_component_information_v1.json');
  const sourceList=f.write('source-list.txt','fixture.cpp\n');
  const admission=path.join(f.root,'.temp/embeddings/runs',policy.run??family,'admission-fixture');
  fs.mkdirSync(admission,{recursive:true});
  const output=path.join(admission,'freeze'),args=['--repo-root',f.root,'--source-list',sourceList,
    '--output',output,'--sdk-proof',SDK];
  const selected=policy.screen?policy.screen:null;
  if(selected) {
    const input=path.join(f.root,'synthetic-inputs');
    for(const name of inputNames(selected))f.write(path.join('synthetic-inputs',name),'Opaque artificial role; never decoded.\n');
    args.push('--masters',selected.join(','),'--input-root',input);
  }
  const entry=policy.fresh||family==='spectral_confirmation_v1'?'prepare_confirmation':policy.information?'prepare':'prepare_screen';
  const record=passed(invoke(family,entry,args));
  assert.equal(record.protocol,policy.protocol);assert.deepEqual(record.masters,selected??policy.masters);
  for(const key of ['model_execution','encoder_training','head_fitting','old_artifacts_replaced'])assert.equal(record[key],false);
  assert.equal(record.migration_implementation.receipt_encoding,JSON_ENCODING);
  assert.equal(record.migration_implementation.equivalence,'not-certified');
  const sourceManifest=fs.readFileSync(path.join(output,'sources.sha256'));
  assert.equal(record.source_fingerprint,sha256(sourceManifest));
  for(const line of sourceManifest.toString().trimEnd().split('\n')) {
    const name=line.slice(66);assert.equal(sha256(fs.readFileSync(path.join(output,'SOURCE',name))),line.slice(0,64));
  }
  const recordName=policy.tasks?'engineering-freeze.json':'freeze.json',frozen=fs.readFileSync(path.join(output,recordName));
  if(selected) {
    const manifest=fs.readFileSync(path.join(output,'inputs.sha256'));
    assert.equal(manifest.toString().trimEnd().split('\n').length,selected.length*3);
    assert.equal(record.inputs_sha256,sha256(manifest));
  }
  const repeated=invoke(family,entry,args);assert.equal(repeated.status,1);assert.match(repeated.stderr,/exists|EEXIST/);
  assert.deepEqual(fs.readFileSync(path.join(output,recordName)),frozen);
});

for(const family of ['unit_relation_screen_v1','fixed_prior_fresh_confirmation_v1'])
  test(`${family} captures metadata sources and a complete artificial inventory through its CLI`,t=>{
    const f=fixture(t),familyPrefix=`${prefix}/${family}/`;
    for(const name of [`${prefix}/node/v1/metadata.mjs`,'code/shared/artifacts/node/v1/files.mjs',
      'code/scripts/node.sh','code/scripts/node-runtime.sh'])f.copy(name);
    for(const entry of fs.readdirSync(path.join(repository,familyPrefix)))if(/\.(?:mjs|sh)$/.test(entry))f.copy(familyPrefix+entry);
    const marker=f.write('.temp/embeddings/runs/fixture/admission-fixture/freeze/fixture.json',{artificial:true});
    const admission=path.dirname(path.dirname(marker));
    const record=passed(invoke(family,'capture_metadata',['--repo-root',f.root,'--admission',admission]));
    const manifest=fs.readFileSync(path.join(admission,'artifact-inventory.sha256'));
    assert.equal(record.inventory_sha256,sha256(manifest));
    assert.equal(record.artifacts,manifest.toString().trimEnd().split('\n').length);
    assert.ok(record.metadata_tools>=5);
    for(const line of manifest.toString().trimEnd().split('\n'))assert.equal(sha256(fs.readFileSync(path.join(admission,line.slice(66)))),line.slice(0,64));
    assert.match(manifest.toString(),/freeze\/fixture\.json/);
    const repeated=invoke(family,'capture_metadata',['--repo-root',f.root,'--admission',admission]);
    assert.equal(repeated.status,1);assert.match(repeated.stderr,/exists|EEXIST/);
    assert.deepEqual(fs.readFileSync(path.join(admission,'artifact-inventory.sha256')),manifest);
  });

for(const [family,policy] of Object.entries(FAMILIES).filter(([,policy])=>policy.screen||policy.fresh))
  test(`${family} summary CLI keeps unsupported artificial cohorts and a negative gate`,t=>{
    const f=fixture(t),masters=policy.fresh?policy.masters:policy.screen,tag=policy.fresh?'RPB-v17.alt-01':'artificial-candidate';
    if(!policy.fresh)f.copy(archivedResults+(policy.unit?'fixed_prior_fresh_confirmation_v1.json':'structured_hard_timing_comparison_v2.json'));
    const run=path.join(f.root,'.temp/embeddings/runs/fixture/admission-fixture/quality');
    const admission=path.dirname(run),relative=path.relative(f.root,run),source='f'.repeat(64),input='a'.repeat(64);
    const method={status:'unsupported',reason:'Artificial unsupported fixture',repetitions:[],
      validation_intact_population:{coverage:.5},validation_deleted_population:{coverage:.5}};
    const cohorts=masters.map(master=>({model_tag:tag,master,points:[0,512].map(updates=>({updates,readouts:{methods:[method]}})),
      training_reconstruction:{standardized_mae:.5},validation_reconstruction:{standardized_mae:.6},
      costs:{},progress:{training_seconds:.001}}));
    for(const master of masters)f.write(`${relative}/seed-${master}/progress.json`,{artificial:true});
    const report=f.write(`${relative}/report.json`,{protocol:policy.protocol,masters,cohorts,source_fingerprint:source,
      inputs_sha256:input,human_card_sha256:'b'.repeat(64),sdk_proof_sha256:'c'.repeat(64),counts:{artificial:true}});
    f.write(`${relative}/saved-check.json`,{status:'passed',report_sha256:sha256(fs.readFileSync(report)),artificial:true});
    const admissionRelative=path.relative(f.root,admission);
    if(policy.fresh) {
      f.write(`${admissionRelative}/freeze/freeze.json`,{source_fingerprint:source,inputs_sha256:input,source_files:1});
      for(const name of ['engineering.log','fresh-data/data-report.json','metadata-sources.sha256','artifact-inventory.sha256'])
        f.write(`${admissionRelative}/${name}`,'Artificial fixture metadata only.\n');
    }
    const summary=path.join(admission,'summary.json'),markdown=path.join(admission,'summary.md');
    const args=['--repo-root',f.root,'--run',run,'--summary',summary,'--report',markdown];
    if(policy.fresh)args.push('--admission',admission);
    const result=passed(invoke(family,policy.fresh?'publish_confirmation':'publish_screen',args));
    assert.equal(result.summary_sha256,sha256(fs.readFileSync(summary)));assert.equal(result.report_sha256,sha256(fs.readFileSync(markdown)));
    const value=JSON.parse(fs.readFileSync(summary,'utf8'));
    assert.equal(value.promotion,false);assert.equal(value.testing_accessed,false);
    assert.equal(value.migration_implementation.receipt_encoding,JSON_ENCODING);
    if(policy.fresh) {
      assert.equal(value.confirmation.continuity.passed,false);assert.equal(value.confirmation.strong_capability.passed,false);
      assert.equal(value.quality.validation_deleted.ridge_mean,null);
    } else {
      assert.equal(value.advancement[tag].passed,false);
      const candidate=value.stages.screen.quality.validation_deleted.find(row=>row.label===tag);
      assert.equal(candidate.ridge_mean,null);assert.equal(candidate.coverage_mean,.5);
      assert.equal(candidate.per_master[0].reason,'Artificial unsupported fixture');
    }
    const frozen=fs.readFileSync(summary);assert.equal(invoke(family,policy.fresh?'publish_confirmation':'publish_screen',args).status,1);
    assert.deepEqual(fs.readFileSync(summary),frozen);
  });
