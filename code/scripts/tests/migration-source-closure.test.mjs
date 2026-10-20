// Source-only build admission: print dependency lists; do not compile or run models.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';

const prep=['training-objective','v7-decoder-calibration','fresh-decoder-replication','saved-native-reliability',
  'frozen-amplitude-transfer','early-mixer-reliability','early-mixer-learning-curve','matched-target-gain',
  'pooled-context','early-mixer-confirmation'];
const prepEntries={'training-objective':'prepare-training-objective-inputs',
  'v7-decoder-calibration':'prepare-v7-decoder-inputs'};
const protocols=['architecture-screen','fixed-prior-relation-screen','partitioned-relation-screen','unit-relation-screen',
  'fixed-prior-fresh-confirmation','two-component-information','multiband-screen','spectral-confirmation'];
const configurable=['configurable-branch','configurable-capacity','configurable-learned-relation','configurable-content-relation'];
const strict=['visible-difference','structured-hard-timing'];
const root='/mityane';
const disposition=JSON.parse(fs.readFileSync(path.join(root,'doc/PYTHON_MIGRATION_DISPOSITION_2026-10-18.json'),'utf8'));
const retiredPython=new Set(disposition.files.filter(row=>row.disposition!=='retain-scientific-python').map(row=>row.path));
const checkedFiles=new Set(),importCache=new Map();
let rootPrints,probeAttempted=false;
function printSources(target) {
  const build=path.join(process.env.TMPDIR,'source-closure-build');
  if(rootPrints)return rootPrints.get(target);
  assert.equal(probeAttempted,false,'the single Make source-list probe already failed');
  probeAttempted=true;
  // One genuine Make invocation prints all twenty-four lists. Its standard
  // debug target boundaries separate the outputs without replacing recipes or
  // introducing a different source list formula. The root include guard reuses
  // its first parse when each original family Makefile includes it again.
  const goals=[...prep,...protocols,...configurable,...strict];
  const makefiles=['Makefile',...[...protocols,...strict].map(name=>
    `code/embeddings/evaluation/protocols/${name.replaceAll('-','_')}_v1/Makefile`)];
  const args=['--no-print-directory','-s','--debug=b',...makefiles.flatMap(name=>['-f',name]),
    ...goals.map(name=>`print-${name}-sources`),`BUILD_DIR=${build}`];
  const result=spawnSync('make',args,{cwd:root,encoding:'utf8',timeout:900000,maxBuffer:2*1024*1024,
    env:{...process.env,LC_ALL:'C'}});
  const logs=fs.mkdtempSync(path.join(process.env.TMPDIR,'source-closure-probe-'));
  fs.writeFileSync(path.join(logs,'stdout.log'),result.stdout??'');
  fs.writeFileSync(path.join(logs,'stderr.log'),result.stderr??'');
  fs.writeFileSync(path.join(logs,'invocation.json'),JSON.stringify({command:'make',args,goals,
    SOURCE_GIT_HEAD:process.env.SOURCE_GIT_HEAD??null,SOURCE_GIT_DIRTY:process.env.SOURCE_GIT_DIRTY??null,
    status:result.status,signal:result.signal,error:result.error?.message??null},null,2)+'\n');
  console.log(`Source-closure Make logs: ${logs}`);
  assert.equal(result.error,undefined);assert.equal(result.status,0,result.stderr);assert.equal(result.stderr,'');
  rootPrints=new Map();let active=null;
  for(const line of result.stdout.split('\n')) {
    const start=line.trim().match(/^Must remake target 'print-(.+)-sources'\.$/);
    if(start){active=start[1];rootPrints.set(active,[]);continue;}
    if(line.trim().startsWith('Successfully remade target ')){active=null;continue;}
    if(active&&line!=='')rootPrints.get(active).push(line);
  }
  assert.deepEqual([...rootPrints.keys()],goals,'all actual root print targets completed');
  return rootPrints.get(target);
}
function moduleImports(filename) {
  if(importCache.has(filename))return importCache.get(filename);
  const source=fs.readFileSync(path.join(root,filename),'utf8'), imports=[];
  for(const match of source.matchAll(/\b(?:import|export)\s+(?:[^;]*?\bfrom\s*)?['"](\.[^'"]+)['"]/g))
    imports.push(path.posix.normalize(path.posix.join(path.posix.dirname(filename),match[1])));
  importCache.set(filename,imports);return imports;
}
function checkClosure(target,_makefile,entry) {
  assert.equal(process.platform,'linux');assert.ok(fs.existsSync('/.dockerenv'));
  assert.equal(process.cwd(),root);assert.ok(process.env.TMPDIR?.startsWith('/opt/mityane-geometrics/'));
  const names=printSources(target);
  assert.ok(names.length>0);assert.deepEqual(names,[...new Set(names)].sort(),'complete sorted unique source roles');
  const declared=new Set(names);
  for(const name of names) {
    assert.ok(!retiredPython.has(name),`${target}: retired Python helper remains in current source closure: ${name}`);
    if(checkedFiles.has(name))continue;
    assert.ok(!path.isAbsolute(name)&&path.posix.normalize(name)===name&&!name.split('/').includes('..'),name);
    const absolute=path.join(root,name),stat=fs.lstatSync(absolute);
    assert.ok(stat.isFile()&&!stat.isSymbolicLink(),name);assert.equal(fs.realpathSync(absolute),absolute,name);
    assert.ok(!name.startsWith('code/embeddings/output/')&&!name.startsWith('.temp/'),name);
    checkedFiles.add(name);
  }
  assert.ok(declared.has(entry),`entry point captured: ${entry}`);
  const pending=names.filter(name=>name.endsWith('.mjs')&&!/(?:_test|\.test)\.mjs$/.test(name)),seen=new Set();
  while(pending.length) {
    const name=pending.pop();if(seen.has(name))continue;seen.add(name);
    for(const dependency of moduleImports(name)) {
      assert.ok(declared.has(dependency),`${target}: ${name} imports uncaptured ${dependency}`);
      pending.push(dependency);
    }
  }
  for(const runtime of ['code/shared/artifacts/node/v1/files.mjs','code/scripts/node.sh','code/scripts/node-runtime.sh'])
    assert.ok(declared.has(runtime),`${target}: shared runtime captured: ${runtime}`);
}
for(const name of prep)test(`root ${name} Make source closure contains every Node import`,()=>
  checkClosure(name,null,`code/embeddings/scripts/${prepEntries[name]??`prepare-${name}`}.mjs`));
for(const name of protocols) {
  const family=name.replaceAll('-','_')+'_v1';
  const stem=name==='fixed-prior-fresh-confirmation'||name==='spectral-confirmation'?'prepare_confirmation':
    name==='two-component-information'?'prepare':'prepare_screen';
  test(`${family} Make source closure contains every Node import`,()=>checkClosure(name,
    `code/embeddings/evaluation/protocols/${family}/Makefile`,`code/embeddings/evaluation/protocols/${family}/${stem}.mjs`));
}
for(const name of configurable) {
  const family=name.replaceAll('-','_')+'_v1';
  test(`${family} root Make source closure contains every Node import`,()=>checkClosure(name,null,
    `code/embeddings/evaluation/protocols/${family}/capsule.mjs`));
}
for(const name of strict) {
  const family=name.replaceAll('-','_')+'_v1';
  test(`${family} strict Make source closure contains every Node import`,()=>checkClosure(name,
    `code/embeddings/evaluation/protocols/${family}/Makefile`,`code/embeddings/evaluation/protocols/${family}/prepare_saved_run.mjs`));
}
