import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {Generator, priorExclusions, run} from './run.mjs';

const directory=path.dirname(new URL(import.meta.url).pathname);
const configuration=JSON.parse(fs.readFileSync(path.join(directory,'configuration.json'),'utf8'));

test('fresh source populations are disjoint from every old source and new split',()=>{
  const exclusions=priorExclusions(configuration,false),used=new Set(exclusions),generators=[];
  let generated=0;
  // Never materialize the prospective scientific assessment population here.
  const counts={training:64,development:16,assessment:24};
  for(const seed of [40031]) {
    const generator=new Generator(seed,used);generators.push(generator);
    for(const split of ['training','development']) generated+=generator.rows(split,counts[split]).length;
  }
  for(const generator of generators) generated+=generator.rows('assessment',counts.assessment).length;
  assert.equal(generated,104);
  assert.equal(used.size,exclusions.length+generated);
  assert.ok(exclusions.length>2000);
});

test('engineering-only complete execution, independent audit and corrupt-record rejection',async t=>{
  assert.equal(process.platform,'linux');
  const binary=process.env.MITYANE_CONTRIBUTION_BINARY;assert.ok(binary);
  const root=fs.mkdtempSync('/opt/mityane-geometrics/operations/counterfactuals/contribution-engineering-');
  const attempt=path.join(root,'attempt-001');
  assert.equal(await run(['--self-check','--binary',binary,'--attempt',attempt]),0);
  const audit=JSON.parse(fs.readFileSync(path.join(attempt,'audit.json'),'utf8'));
  assert.equal(audit.valid,true);
  await t.test('exclusive destination preserves completed engineering evidence',async()=>{
    const original=fs.readFileSync(path.join(attempt,'final.json'));
    await assert.rejects(run(['--self-check','--binary',binary,'--attempt',attempt]));
    assert.deepEqual(fs.readFileSync(path.join(attempt,'final.json')),original);
  });
  await t.test('independent auditor rejects one altered successful contrast',()=>{
    const corrupted=path.join(root,'corrupted-copy');fs.cpSync(attempt,corrupted,{recursive:true,errorOnExist:true});
    const filename=path.join(corrupted,'seed-40031/assessment-pairs.tsv');
    const lines=fs.readFileSync(filename,'utf8').trimEnd().split('\n'),row=lines[1].split('\t');
    row[10]=String(Number(row[10])+1);lines[1]=row.join('\t');fs.writeFileSync(filename,lines.join('\n')+'\n');
    const result=spawnSync('python3',['-B',path.join(directory,'audit.py'),corrupted,path.join(root,'corrupted-audit.json')],{encoding:'utf8',timeout:300000,env:{...process.env,PYTHONDONTWRITEBYTECODE:'1'}});
    assert.equal(result.status,1,`${result.stdout}\n${result.stderr}`);
    const rejected=JSON.parse(fs.readFileSync(path.join(root,'corrupted-audit.json'),'utf8'));assert.equal(rejected.valid,false);
    assert.ok(rejected.audit.errors.length>0);
  });
  console.log(JSON.stringify({engineering_attempt:attempt,excluded_from_scientific_results:true}));
});
