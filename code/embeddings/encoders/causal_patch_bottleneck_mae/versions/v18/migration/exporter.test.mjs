// Copyright (c) 2026 Waajacu. MIT; see LICENSE at repository root.
// Wrapper routing and independent recipient checks. Generic integrity cases are
// owned by export-cpb-family.test.mjs; no numerical code or datasets execute here.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {V18_PROFILE} from '../../../../../scripts/export-cpb-v18.mjs';

const repository=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../../../../../../..');
const wrapper='code/embeddings/scripts/export-cpb-v18.mjs';
const dependency='code/embeddings/scripts/export-cpb-family.mjs';
function fixture(t){
  assert.equal(process.platform,'linux','run in the managed Linux container');
  assert(fs.existsSync('/.dockerenv'),'managed container required');
  const temporary=fs.mkdtempSync(path.join(os.tmpdir(),'cpb-v18-source-export-'));
  t.after(()=>{
    assert.equal(path.dirname(temporary),path.resolve(os.tmpdir()));
    assert(path.basename(temporary).startsWith('cpb-v18-source-export-'));
    fs.rmSync(temporary,{recursive:true});
  });
  const root=path.join(temporary,'origin'),output=path.join(temporary,'bundle');
  const put=(name,bytes)=>{const absolute=path.join(root,name);fs.mkdirSync(path.dirname(absolute),{recursive:true});fs.writeFileSync(absolute,bytes);};
  for(const name of [wrapper,dependency])put(name,fs.readFileSync(path.join(repository,name)));
  put('include/v18.h','#include "core.h"\n// synthetic mask-recipe interface\n');
  put('include/core.h','#pragma once\n// synthetic architecture dependency\n');
  put('LICENSE','Synthetic source license bytes\r\n');
  const profile={schema:'cpb-family-source-package-v1',family:'CPB-v18',include_roots:['include'],files:[
    {path:'include/v18.h',role:'versioned-recipe-api'},{path:'include/core.h',role:'preserved-architecture'},
    {path:'LICENSE',role:'license'},{path:wrapper,role:'source-export-tool'},{path:dependency,role:'source-export-tool'},
    {path:V18_PROFILE,role:'versioned-export-profile'}]};
  put(V18_PROFILE,`${JSON.stringify(profile)}\n`);
  return {root,output,temporary,cli:path.join(root,wrapper)};
}
function run(script,args,expected=0){
  const result=spawnSync(process.execPath,[script,...args],{encoding:'utf8',timeout:30000});
  assert.equal(result.signal,null,'normal handled process exit required');
  assert.ifError(result.error);assert.equal(result.status,expected,result.stderr);return result;
}
test('V18 wrapper chooses its own explicit inventory and recipient verifies after relocation',t=>{
  const f=fixture(t),exported=JSON.parse(run(f.cli,['--output',f.output]).stdout);
  const manifest=JSON.parse(fs.readFileSync(path.join(f.output,'manifest.json')));
  assert.equal(manifest.profile.path,V18_PROFILE);assert.equal(exported.files,6);assert.equal(exported.quotedIncludesChecked,1);
  const selectedProfile=JSON.parse(fs.readFileSync(path.join(f.output,'sources',V18_PROFILE)));
  assert.equal(selectedProfile.family,'CPB-v18');
  const relocated=path.join(f.temporary,'recipient');fs.renameSync(f.output,relocated);
  fs.renameSync(f.root,path.join(f.temporary,'origin-unavailable'));
  const delivered=path.join(relocated,'sources',wrapper);
  const verified=JSON.parse(run(delivered,['--verify',relocated,'--expected-manifest-sha256',exported.manifestSha256]).stdout);
  assert.equal(verified.sourceClosureSha256,exported.sourceClosureSha256);assert.equal(verified.files,6);
  const refusal=run(delivered,['--verify',relocated,'--expected-manifest-sha256','0'.repeat(64)],1);
  assert.match(refusal.stderr,/manifest SHA256 mismatch/);
});
test('V18 CLI refuses mixed modes, missing independent pin, duplicate or unknown options',t=>{
  const f=fixture(t);
  for(const args of [[],['--verify',f.output],['--output',f.output,'--verify',f.output],
    ['--output',f.output,'--output',f.output],['--unknown','x']]){
    assert.match(run(f.cli,args,1).stderr,/Usage:/);
  }
  assert(!fs.existsSync(f.output));assert.match(run(f.cli,['--help']).stdout,/export-cpb-v18/);
});
