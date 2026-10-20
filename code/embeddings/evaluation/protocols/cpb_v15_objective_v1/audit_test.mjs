// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
// Container-only synthetic saved-array checks. Preserve every fixture/result.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {randomUUID} from 'node:crypto';
import {spawnSync} from 'node:child_process';

function native(name){const value=process.env[name];assert.ok(value?.startsWith('/opt/mityane-geometrics/'),`${name} must name the reviewed managed binary`);assert.equal(fs.realpathSync(value),value);assert.ok(fs.statSync(value).isFile());return value;}
function execute(binary,args){return spawnSync(binary,args,{encoding:'utf8',maxBuffer:4*1024*1024,timeout:120000,env:{...process.env,LC_ALL:'C'}});}
function near(actual,expected){assert.ok(Number.isFinite(actual)&&Math.abs(actual-expected)<=2e-8*(1+Math.abs(expected)),`${actual} differs from independent artificial expectation ${expected}`);}
function waveformExpected(){
  const rows=[[[0,1],[4]],[[8,9,10]]],fr=Math.fround;let mae=0,nll=0,huber=0;
  for(const channels of rows){let rm=0,rn=0,rh=0;for(const cells of channels){let cm=0,cn=0,ch=0;for(const i of cells){
    const mean=fr(fr((i%5)-2)*fr(.6)),target=fr(fr(i%3)*fr(.2)),variance=fr(.5+fr(.25*fr(i%3)));
    const error=mean-target,a=Math.abs(error);cm+=a;cn+=.5*(Math.log(variance)+error*error/variance);ch+=a<=1?.5*error*error:a-.5;
  }rm+=cm/cells.length;rn+=cn/cells.length;rh+=ch/cells.length;}mae+=rm/channels.length;nll+=rn/channels.length;huber+=rh/channels.length;}
  return {mae:mae/2,nll:nll/2,huber:huber/2};
}
test('independent saved-fit/waveform auditor passes artificial positives and retains all negative failures',()=>{
  assert.equal(process.platform,'linux');assert.ok(fs.existsSync('/.dockerenv'));assert.equal(fs.realpathSync('/mityane'),'/mityane');
  const auditor=native('CPB_V15_AUDIT_BINARY'),generator=native('CPB_V15_AUDIT_FIXTURE_BINARY');
  const parent=path.dirname(auditor);const output=path.join(parent,`artificial-audit-${randomUUID()}`);
  assert.equal(fs.realpathSync(parent),parent);assert.ok(!fs.existsSync(output));
  const generated=execute(generator,['--output',output]);
  assert.equal(generated.error,undefined,generated.error?.message);assert.equal(generated.status,0,generated.stderr);
  const manifest=JSON.parse(fs.readFileSync(path.join(output,'fixture-manifest.json'),'utf8'));
  assert.equal(manifest.kind,'cpb_v15_independent_audit_artificial_fixtures_v1');assert.equal(manifest.real_dataset_reads,0);
  assert.ok(manifest.cases.length>=30);let passed=0,refused=0;
  for(const row of manifest.cases){
    assert.equal(fs.realpathSync(row.plan),row.plan);assert.ok(row.plan.startsWith(output+'/'));
    const receipt=path.join(path.dirname(row.plan),'audit.json'),actual=execute(auditor,['--plan',row.plan,'--output',receipt]);
    fs.writeFileSync(path.join(path.dirname(row.plan),'audit.stdout'),actual.stdout??'',{flag:'wx'});
    fs.writeFileSync(path.join(path.dirname(row.plan),'audit.stderr'),actual.stderr??'',{flag:'wx'});
    assert.equal(actual.error,undefined,`${row.name}: ${actual.error?.message}`);
    assert.ok(fs.existsSync(receipt),`${row.name}: retained pass/failure receipt missing`);
    const result=JSON.parse(fs.readFileSync(receipt,'utf8'));
    assert.equal(result.kind,'cpb_v15_independent_saved_fit_audit_v1');assert.equal(result.passed,row.expect_pass,`${row.name}: ${actual.stderr}`);
    if(row.expect_pass){++passed;assert.equal(actual.status,0,`${row.name}: ${actual.stderr}`);assert.equal(result.array_sets,1);
      assert.equal(result.model_forwards,0);assert.equal(result.head_fits,0);assert.equal(result.optimizer_replays,0);
      if(row.name.startsWith('regression')){const entry=result.entries[0];assert.equal(entry.actual_seed_decimal,'18446744073709551615');assert.equal(entry.tasks.length,4);
        // Closed-form t0 ridge prediction is10 for each of three valid probe
        // rows, whose truths are11,12,13. The fourth feature row abstains.
        near(entry.tasks[0].ridge.mae,2);near(entry.tasks[0].ridge.rmse,Math.sqrt(14/3));near(entry.tasks[0].ridge.r2,-6);
        assert.deepEqual(entry.tasks.map(task=>task.ridge.valid),[3,3,2,3]);}
      if(row.name.startsWith('classification')){assert.ok(result.entries[0].exact_saved_logit_ties>=1);near(result.entries[0].ridge_accuracy,1);near(result.entries[0].coverage,.75);}
      if(row.name==='classification-near-tie')assert.equal(result.entries[0].neural_accuracy,1/3);
      if(row.name.startsWith('objective')||row.name.startsWith('fixed_query')){const entry=result.entries[0],expected=row.name==='objective-negative-nll'?{mae:0,nll:Math.log(.25)/2,huber:0}:waveformExpected();assert.equal(entry.valid_examples,2);assert.equal(entry.target_cells,6);assert.equal(entry.gradient_reconstruction,false);near(entry.mae,expected.mae);near(entry.gaussian_nll,expected.nll);near(entry.mean_huber,expected.huber);}
    }else{++refused;assert.notEqual(actual.status,0,row.name);assert.equal(result.status,'failed');assert.ok(result.error);}
  }
  assert.ok(passed>=6&&refused>=25);
  // The auditor must preserve an existing result even on a repeated request.
  const first=manifest.cases[0],receipt=path.join(path.dirname(first.plan),'audit.json'),before=fs.readFileSync(receipt);
  const repeated=execute(auditor,['--plan',first.plan,'--output',receipt]);assert.notEqual(repeated.status,0);assert.deepEqual(fs.readFileSync(receipt),before);
  fs.writeFileSync(path.join(output,'driver-result.json'),JSON.stringify({passed:true,positive_cases:passed,negative_cases:refused,artifacts_retained:true})+'\n',{flag:'wx'});
});
