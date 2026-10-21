// Copyright (c) 2026 Waajacu. See LICENSE.
// Artificial metadata only; never opens project tensors or starts recovery.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {parseCSV,mergeCSV} from './reconcile.mjs';
import {admitFiles,writeNew} from '../../../../shared/artifacts/node/v1/files.mjs';

const HEADERS={
  'training.csv':'attempt,update,training_objective_F64,eligible_examples,target_cells,nonzero_gradient_tensors,changed_tensors,row_seed,mask_seed,anchors',
  'temporal-training.csv':'attempt,update,base_nll,temporal_loss,weighted_temporal,total_loss,lag1_loss,lag4_loss,lag16_loss,lag1_pairs,lag4_pairs,lag16_pairs,lag1_supported,lag4_supported,lag16_supported'
};
const row=(name,u)=>name==='training.csv'?
  `${u-1},${u},0.125,8,6912,75,75,18446744073709551615,9007199254740993,"1704953279999;1705735259999;1705500659999;1704658919999;1705682999999;1704177839999;1704122939999;1704638279999"`:
  `${u-1},${u},-0.125,0.25,0.025,-0.1,0.125,0.25,0.375,6210,4104,1224,1,1,1`;
function csv(name,first,last,{fragment='',newline=true}={}){
  return Buffer.from(`${HEADERS[name]}\n${Array.from({length:last-first+1},(_,i)=>row(name,first+i)).join('\n')}${newline?'\n':''}${fragment}`);
}
const suffix=name=>csv(name,513,2048);

test('preserved prefix plus suffix has each of 2048 logical updates exactly once',()=>{
  for(const name of Object.keys(HEADERS)){
    const last=name==='training.csv'?553:565,fragment=row(name,last+1).slice(0,43);
    const actual=mergeCSV(csv(name,1,last,{fragment}),suffix(name),name);
    const decoded=parseCSV(actual.merged,name,{first:1,last:2048});
    assert.equal(decoded.rows.length,2048);
    assert.equal(actual.comparison.complete_replayed_overlap_rows,last-512);
    assert.equal(actual.comparison.fragment_exact_replayed_line_prefix,true);
    assert.equal(actual.comparison.fragment_certifies_completed_update,false);
    assert.equal(actual.comparison.tolerance_used,false);
    assert.equal(actual.comparison.binary64_float_bits_equal,true);
  }
});
test('unsafe U64 seeds remain exact strings',()=>{
  const parsed=parseCSV(csv('training.csv',1,1),'training.csv');
  assert.equal(parsed.rows[0].fields[7],'18446744073709551615');
  assert.equal(parsed.rows[0].fields[8],'9007199254740993');
});
test('mismatching complete scalar aborts without tolerance',()=>{
  const original=csv('temporal-training.csv',1,565).toString().replace(row('temporal-training.csv',540),row('temporal-training.csv',540).replace('0.25','0.25000000000000006'));
  assert.throws(()=>mergeCSV(Buffer.from(original),suffix('temporal-training.csv'),'temporal-training.csv'),/differs; no rescue/);
});
test('equal floating value with different scalar lexeme still refuses byte drift',()=>{
  const original=csv('training.csv',1,553).toString().replace(row('training.csv',530),row('training.csv',530).replace('0.125','0.1250'));
  assert.throws(()=>mergeCSV(Buffer.from(original),suffix('training.csv'),'training.csv'),/differs; no rescue/);
});
test('mismatching U64 stream string aborts',()=>{
  const original=csv('training.csv',1,553).toString().replace(row('training.csv',540),row('training.csv',540).replace('9007199254740993','9007199254740992'));
  assert.throws(()=>mergeCSV(Buffer.from(original),suffix('training.csv'),'training.csv'),/differs; no rescue/);
});
test('mismatching partial final field aborts rather than discarding a fragment',()=>{
  const name='temporal-training.csv',fragment=row(name,566).slice(0,45).replace('-0.125','-0.126');
  assert.throws(()=>mergeCSV(csv(name,1,565,{fragment}),suffix(name),name),/fragment differs; no rescue/);
});
test('a torn original row is preserved and checked without certifying completion',()=>{
  const name='training.csv',fragment=row(name,554).slice(0,-19);
  const result=mergeCSV(csv(name,1,553,{fragment}),suffix(name),name);
  assert.equal(result.comparison.original_last_complete_update,553);
  assert.equal(result.comparison.complete_replayed_overlap_rows,41);
  assert.equal(result.comparison.original_fragment_bytes,Buffer.byteLength(fragment));
  assert.equal(result.comparison.fragment_certifies_completed_update,false);
});
test('suffix must have complete final row and exactly 1536 successful rows',()=>{
  assert.throws(()=>mergeCSV(csv('training.csv',1,553),csv('training.csv',513,2048,{newline:false}),'training.csv'),/end with newline/);
  assert.throws(()=>mergeCSV(csv('training.csv',1,553),csv('training.csv',513,2047),'training.csv'),/final complete CSV update/);
});
test('duplicate, skipped and reset logical counters refuse',()=>{
  for(const replacement of ['511,513','513,514','0,1']){
    const raw=suffix('training.csv').toString().replace('512,513',replacement);
    assert.throws(()=>parseCSV(Buffer.from(raw),'training.csv',{first:513,last:2048}),/attempt\/update/);
  }
});
test('header, UTF-8 line encoding and column contracts are exact',()=>{
  assert.throws(()=>parseCSV(Buffer.from('bad\n'),'training.csv'),/header/);
  assert.throws(()=>parseCSV(Buffer.from(csv('training.csv',1,1).toString().replaceAll('\n','\r\n')),'training.csv'),/LF UTF-8/);
  assert.throws(()=>parseCSV(Buffer.from(csv('temporal-training.csv',1,1).toString().replace(',1,1,1\n',',1,1\n')),'temporal-training.csv'),/columns/);
});
test('nonfinite scalars, out-of-range identities and non-Boolean support refuse',()=>{
  for(const [name,from,to,error] of [
    ['training.csv','0.125','1e309',/finite/],
    ['training.csv','18446744073709551615','18446744073709551616',/U64/],
    ['temporal-training.csv',',1,1,1\n',',1,1,2\n',/Boolean/]
  ])assert.throws(()=>parseCSV(Buffer.from(csv(name,1,1).toString().replace(from,to)),name),error);
});
test('fragment counters must describe the immediately following logical row',()=>{
  assert.throws(()=>parseCSV(csv('training.csv',1,553,{fragment:'555,556,0.125'}),'training.csv',{partial:true}),/fragment has next logical counters/);
});
test('metadata publication primitives reject overwrite and redirected or aliased evidence',()=>{
  assert.equal(process.platform,'linux','container-only metadata fixture');
  const build='/opt/mityane-geometrics/embeddings/build/cpb-v16-temporal-001';
  const directory=fs.mkdtempSync(path.join(build,'recovery-bridge-metadata-fixture-'));
  const original=path.join(directory,'original.csv'),hardAlias=path.join(directory,'hard-alias.csv'),symbolicAlias=path.join(directory,'symbolic-alias.csv');
  writeNew(original,'preserve these artificial bytes\n');
  assert.throws(()=>writeNew(original,'replacement forbidden\n'),/EEXIST/);
  assert.equal(fs.readFileSync(original,'utf8'),'preserve these artificial bytes\n');
  fs.linkSync(original,hardAlias);
  assert.throws(()=>admitFiles([original,hardAlias],{root:directory}),/aliased admitted file/);
  assert.throws(()=>admitFiles([original],{root:directory,singleLink:true}),/single-link/);
  fs.symlinkSync(original,symbolicAlias);
  assert.throws(()=>admitFiles([symbolicAlias],{root:directory}),/regular canonical/);
  // Preserve this tiny named fixture in the container build tree for inspection.
});
