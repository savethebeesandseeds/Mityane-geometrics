// Artificial metadata fixtures. Run in the managed container, never on Windows.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {mean,inputNames,FAMILIES,rowsFor,aggregate,prepare,publish} from './metadata.mjs';
import {sha256} from '../../../../../shared/artifacts/node/v1/files.mjs';
import {status} from './gate_status.mjs';
const retired=[['unit_relation_screen_v1','record_screen'],['partitioned_relation_screen_v1','record_versions'],
  ['fixed_prior_fresh_confirmation_v1','record_confirmation'],['two_component_information_v1','publish'],
  ['multiband_screen_v1','publish'],['spectral_confirmation_v1','publish']];
test('metadata averages retain exact binary rational summation',()=>{
  assert.equal(mean([1e16,1,-1e16]),1/3);
  assert.equal(mean([Number.MIN_VALUE,Number.MIN_VALUE]),Number.MIN_VALUE);
  assert.equal(mean([.1,.2]),.15000000000000002);
  assert.equal(mean([.9,null]),null);
  assert.throws(()=>mean([]));assert.throws(()=>mean([NaN]));
});
test('closed source protocols retain complete cohort/task/view roles',()=>{
  assert.equal(inputNames(FAMILIES.multiband_screen_v1.masters,FAMILIES.multiband_screen_v1.tasks).length,12);
  assert.equal(inputNames(FAMILIES.spectral_confirmation_v1.masters,FAMILIES.spectral_confirmation_v1.tasks).length,18);
  assert.equal(inputNames(FAMILIES.fixed_prior_fresh_confirmation_v1.masters).length,15);
});
test('unsupported method retains coverage and missing scores',()=>{
  const cohort={master:1,points:[{updates:512,readouts:{methods:[{status:'unsupported',reason:'frozen abstention',repetitions:[],validation_deleted_population:{coverage:.5}}]}}]};
  const rows=rowsFor([cohort],512,'validation_deleted'),summary=aggregate('candidate',rows);
  assert.equal(summary.ridge_mean,null);assert.equal(summary.tiny_secondary_mean,null);
  assert.equal(summary.coverage_mean,.5);assert.equal(rows[0].reason,'frozen abstention');
});
test('gate status keeps failure distinct from malformed metadata',()=>{
  assert.equal(status({gate:{passed:true}}),0);assert.equal(status({gate:{passed:false}}),3);
  assert.throws(()=>status({gate:{passed:1}}));assert.throws(()=>status({}));
});
for(const[family,name]of retired)test(`retired ${family}/${name} preserves refusal without import effects`,async()=>{
  const prior=process.exitCode;
  const module=await import(`../../${family}/${name}.mjs`);
  assert.equal(process.exitCode,prior);
  assert.equal(module.refusal,`Retired historical publisher: original source is in .archive/embeddings/pre-cleanup-2026-10-18/code/evaluation/protocols/${family}/${name}.py. Historical registries are immutable; use a separately versioned publication protocol for new work.`);
});
function fixture(t) {
  assert.equal(process.platform,'linux');assert.equal(fs.existsSync('/.dockerenv'),true,'managed container fixtures only');
  const base='/opt/mityane-geometrics/python-migration-fixtures';fs.mkdirSync(base,{recursive:true});
  const repo=fs.mkdtempSync(path.join(base,'protocol-metadata-'));
  t.after(()=>fs.rmSync(repo,{recursive:true,force:true}));
  const output=path.join(repo,'.temp/embeddings/runs/rpb-multiband-screen/admission-fixture/freeze');
  const input=path.join(path.dirname(output),'quality-data');fs.mkdirSync(output,{recursive:true});fs.mkdirSync(input);
  const write=(filename,value)=>{fs.mkdirSync(path.dirname(filename),{recursive:true});fs.writeFileSync(filename,typeof value==='string'?value:JSON.stringify(value)+'\n');};
  const source='// artificial source fixture\n',manifest=`${sha256(source)}  entry.mjs\n`;
  write(path.join(repo,'entry.mjs'),source);write(path.join(output,'SOURCE/entry.mjs'),source);
  write(path.join(output,'sources.sha256'),manifest);
  const sourceID=sha256(manifest),cardSHA='0'.repeat(64),names=inputNames(FAMILIES.multiband_screen_v1.masters,FAMILIES.multiband_screen_v1.tasks);
  const files=names.map((name,index)=>{const bytes=`opaque fixture ${index}\n`;write(path.join(input,name),bytes);return{path:name,bytes:Buffer.byteLength(bytes),sha256:sha256(bytes)};});
  const report={protocol:'multiband-screen-v1',data_recipe:'two-component-v1',source_fingerprint:sourceID,human_card_sha256:cardSHA,
    masters:[920903,920904],tasks:['slow_lag_sign','component_balance'],generator_calls:4,archive_count:12,encoder_calls:0,information_fit_calls:0,files};
  write(path.join(output,'engineering-freeze.json'),{protocol:report.protocol,source_fingerprint:sourceID,card_sha256:cardSHA});
  function reports(){write(path.join(input,'data-report.json'),report);write(path.join(input,'data-complete.json'),{status:'complete',protocol:report.protocol,
    source_fingerprint:sourceID,human_card_sha256:cardSHA,masters:report.masters,archive_count:12,data_report_sha256:sha256(fs.readFileSync(path.join(input,'data-report.json')))});}
  reports();return{repo,output,input,files,report,reports};
}
test('late generated-role size failure precedes the first payload hash',t=>{
  const f=fixture(t);f.report.files.at(-1).bytes+=1;f.reports();
  const payloads=new Set(f.files.map(row=>path.join(f.input,row.path)));let opened=0;const originalOpen=fs.openSync;
  t.mock.method(fs,'openSync',(file,...args)=>{if(payloads.has(file))opened++;return originalOpen(file,...args);});
  assert.throws(()=>prepare('multiband_screen_v1',{'repo-root':f.repo,output:f.output,'bind-input-root':f.input}),/byte sizes changed/);
  assert.equal(opened,0);assert.equal(fs.existsSync(path.join(f.output,'inputs.sha256')),false);
});
test('generated data bind retains the complete twelve-role manifest',t=>{
  const f=fixture(t),result=prepare('multiband_screen_v1',{'repo-root':f.repo,output:f.output,'bind-input-root':f.input});
  assert.equal(result.input_files,12);assert.equal(result.input_root,f.input);
  assert.equal(fs.readFileSync(path.join(f.output,'inputs.sha256'),'utf8').split('\n').filter(Boolean).length,12);
  assert.equal(result.inputs_sha256,sha256(fs.readFileSync(path.join(f.output,'inputs.sha256'))));
});
test('publication paths are rejected before report access outside ignored runs',t=>{
  const f=fixture(t),allowed=path.join(f.repo,'.temp/embeddings/runs/new-summary.json');
  for(const [summary,report]of [[path.join(f.repo,'doc/summary.json'),allowed],[allowed,path.join(f.repo,'.archive/report.md')]]){
    assert.throws(()=>publish('architecture_screen_v1',{'repo-root':f.repo,run:[path.join(f.repo,'missing-run')],summary,report}),/current ignored run root/);
    assert.equal(fs.existsSync(path.join(f.repo,'doc')),false);assert.equal(fs.existsSync(path.join(f.repo,'.archive')),false);
  }
});
