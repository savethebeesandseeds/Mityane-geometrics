// Bounded CLI fixtures: no producer, tensor decoder, fitting or research runner.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {main as capsuleCLI} from './capsule_cli.mjs';
import {FAMILIES, inputNames} from './metadata.mjs';
import {sha256} from '../../../../../shared/artifacts/node/v1/files.mjs';

const protocols = '/mityane/code/embeddings/evaluation/protocols';
const retired = [['unit_relation_screen_v1','record_screen'], ['partitioned_relation_screen_v1','record_versions'],
  ['fixed_prior_fresh_confirmation_v1','record_confirmation'], ['two_component_information_v1','publish'],
  ['multiband_screen_v1','publish'], ['spectral_confirmation_v1','publish']];
function invoke(relative, args = []) {
  const result = spawnSync(process.execPath, [path.join(protocols, relative), ...args],
    {cwd:'/mityane', encoding:'utf8', timeout:10000, maxBuffer:1024*1024});
  assert.equal(result.error, undefined); assert.equal(result.signal, null);
  return result;
}
function fixture(t) {
  assert.equal(process.platform,'linux'); assert.equal(fs.existsSync('/.dockerenv'),true);
  assert.equal(process.cwd(),'/mityane');
  const parent = process.env.TMPDIR;
  assert.ok(parent?.startsWith('/opt/mityane-geometrics/'));
  const root = fs.mkdtempSync(path.join(parent,'protocol-cli-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  const write = (relative,value) => {
    const target=path.join(root,relative); fs.mkdirSync(path.dirname(target),{recursive:true});
    fs.writeFileSync(target,typeof value==='string'?value:JSON.stringify(value)+'\n'); return target;
  };
  return {root,write};
}

for (const [family,name] of retired) test(`retired CLI ${family}/${name} refuses before parsing options`,()=>{
  const result=invoke(`${family}/${name}.mjs`,['--registry','/no-such-registry','--unknown']);
  assert.equal(result.status,1); assert.equal(result.stdout,'');
  assert.equal(result.stderr,`Retired historical publisher: original source is in .archive/embeddings/pre-cleanup-2026-10-18/code/evaluation/protocols/${family}/${name}.py. Historical registries are immutable; use a separately versioned publication protocol for new work.\n`);
});

test('all eight metadata preparation CLIs reject unknown options before reading inputs',()=>{
  for (const [family,policy] of Object.entries(FAMILIES)) {
    const name=policy.fresh||family==='spectral_confirmation_v1'?'prepare_confirmation':
      policy.information?'prepare':'prepare_screen';
    const result=invoke(`${family}/${name}.mjs`,['--not-a-declared-option','/no-such-file']);
    assert.equal(result.status,1,family); assert.equal(result.stdout,'');
    assert.match(result.stderr,/Unknown option '--not-a-declared-option'/);
  }
});

for (const family of ['configurable_branch_v1','configurable_capacity_v1',
  'configurable_learned_relation_v1','configurable_content_relation_v1'])
  test(`${family} CLI rejects tensor bind without opening a capsule`,()=>{
    const result=invoke(`${family}/capsule.mjs`,['bind','/no-such-capsule']);
    assert.equal(result.status,1); assert.equal(result.stdout,'');
    assert.match(result.stderr,/metadata-only capsule mode required/);
    assert.doesNotMatch(result.stderr,/ENOENT/);
  });

test('capsule CLI rebinds recorded physical paths and dispatches inert callbacks only',async t=>{
  const f=fixture(t),build=path.join(f.root,'build');fs.mkdirSync(build);
  const mapping=[['rpb-v20-capacity','original'],['rpb-v20-branches','branch'],['rpb-v21-learned-relation','previous']]
    .map(([fragment,current])=>({recorded_path:`/embedding/output/runs/${fragment}/admission-preserved`,current_path:path.join(f.root,current)}));
  f.write('capsule/freeze/receipt.json',{build_directory:build,
    migration_implementation:{historical_path_mapping:mapping}});
  const output=[];t.mock.method(process.stdout,'write',value=>{output.push(value);return true;});
  for (const mode of ['finish','verify-admission','bind-quality','admit-third']) {
    let options=null,called=0;
    const method={'verify-admission':'verifyAdmission','bind-quality':'bindQuality','admit-third':'admitThird'}[mode]??mode;
    const result=await capsuleCLI('configurable_content_relation_v1',value=>{
      options=value;
      return {[method]:root=>{called++;assert.equal(root,path.join(f.root,'capsule'));return {mode,inert:true};}};
    },[mode,path.join(f.root,'capsule')]);
    assert.deepEqual(result,{mode,inert:true}); assert.equal(called,1);
    assert.equal(options.repositoryRoot,'/mityane'); assert.equal(options.buildDirectory,build);
    assert.equal(options.originalCapsule,path.join(f.root,'original'));
    assert.equal(options.branchCapsule,path.join(f.root,'branch'));
    assert.equal(options.previousCapsule,path.join(f.root,'previous'));
    assert.equal(options.commandOutput,undefined,'no executable callback for completion/status modes');
    assert.equal(options.sourceList,undefined);
  }
  assert.equal(output.length,4); output.forEach(value=>assert.equal(JSON.parse(value).inert,true));
});

for (const family of ['multiband_screen_v1','spectral_confirmation_v1'])
  test(`${family} CLI binds its complete synthetic generated-role matrix`,t=>{
    const f=fixture(t),policy=FAMILIES[family],relative=`.temp/embeddings/runs/${policy.run}/admission-fixture`;
    const freeze=path.join(f.root,relative,'freeze'),input=path.join(f.root,relative,'quality-data');
    const source='// inert engineering source\n',sourceManifest=`${sha256(source)}  entry.mjs\n`;
    f.write('entry.mjs',source); f.write(`${relative}/freeze/SOURCE/entry.mjs`,source);
    f.write(`${relative}/freeze/sources.sha256`,sourceManifest);
    const sourceID=sha256(sourceManifest),cardSHA='0'.repeat(64),names=inputNames(policy.masters,policy.tasks);
    const files=names.map((name,index)=>{
      const bytes=`Opaque artificial archive role ${index}; never decoded.\n`;
      f.write(`${relative}/quality-data/${name}`,bytes);
      return {path:name,bytes:Buffer.byteLength(bytes),sha256:sha256(bytes)};
    });
    const report={protocol:policy.protocol,data_recipe:'two-component-v1',source_fingerprint:sourceID,
      human_card_sha256:cardSHA,masters:policy.masters,tasks:policy.tasks,generator_calls:policy.masters.length*2,
      archive_count:names.length,encoder_calls:0,information_fit_calls:0,files};
    const reportPath=f.write(`${relative}/quality-data/data-report.json`,report);
    f.write(`${relative}/quality-data/data-complete.json`,{status:'complete',protocol:policy.protocol,
      source_fingerprint:sourceID,human_card_sha256:cardSHA,masters:policy.masters,archive_count:names.length,
      data_report_sha256:sha256(fs.readFileSync(reportPath))});
    f.write(`${relative}/freeze/engineering-freeze.json`,{protocol:policy.protocol,source_fingerprint:sourceID,card_sha256:cardSHA});
    const entry=family==='spectral_confirmation_v1'?'prepare_confirmation':'prepare_screen';
    const args=['--repo-root',f.root,'--output',freeze,'--bind-input-root',input];
    const result=invoke(`${family}/${entry}.mjs`,args);
    assert.equal(result.status,0,result.stderr);assert.equal(result.stderr,'');
    const receipt=JSON.parse(result.stdout);assert.equal(receipt.input_files,names.length);
    const manifest=fs.readFileSync(path.join(freeze,'inputs.sha256'),'utf8');
    assert.equal(manifest,files.map(row=>`${row.sha256}  ${row.path}\n`).join(''));
    assert.equal(receipt.inputs_sha256,sha256(manifest));
    const before=fs.readFileSync(path.join(freeze,'freeze.json'));
    const repeat=invoke(`${family}/${entry}.mjs`,args);
    assert.equal(repeat.status,1);assert.match(repeat.stderr,/EEXIST|exists/);
    assert.deepEqual(fs.readFileSync(path.join(freeze,'freeze.json')),before);
  });

test('gate status CLI distinguishes a valid negative verdict from malformed metadata',t=>{
  const f=fixture(t);
  for (const [name,value,status] of [['passed',{gate:{passed:true}},0],['failed',{gate:{passed:false}},3],
    ['malformed',{gate:{passed:'false'}},1]]) {
    const result=invoke('node/v1/gate_status.mjs',[f.write(`${name}.json`,value)]);
    assert.equal(result.status,status);assert.equal(result.stdout,'');
    if(status===1)assert.match(result.stderr,/typed gate status required/);else assert.equal(result.stderr,'');
  }
});

for (const family of ['visible_difference_v1','structured_hard_timing_v1'])
  test(`${family} saved-run CLI keeps a closed action and option set`,()=>{
    const result=invoke(`${family}/prepare_saved_run.mjs`,['unknown-action']);
    assert.equal(result.status,1);assert.equal(result.stdout,'');assert.match(result.stderr,/closed .* metadata action/);
    const duplicate=invoke(`${family}/prepare_saved_run.mjs`,['self-test','--repo-root','/mityane','--repo-root','/mityane']);
    assert.equal(duplicate.status,1);assert.match(duplicate.stderr,/unique valued option/);
  });
