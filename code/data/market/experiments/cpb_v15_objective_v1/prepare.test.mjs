// Copyright (c) 2026 Waajacu. All rights reserved. See LICENSE.
import test from 'node:test';
import assert from 'node:assert/strict';
import {deflateRawSync} from 'node:zlib';
import {checksumDigest, crc32, extractCSV, GEOMETRY, METADATA, SYMBOLS, validateDeclaration, validateParityReceipt,
  prepareMetadata, importConfirmation} from './prepare.mjs';
const member = 'BTCUSDT-1m-2024-02.csv';
const payload = Buffer.from('1706745600000,1.25,2.5,1,2,10,1706745659999,20,3,4,5,0\n');
function zip(data = payload, {name = member, method = 8, descriptor = false} = {}) {
  const nameBytes = Buffer.from(name), packed = method === 0 ? data : deflateRawSync(data);
  const local = Buffer.alloc(30), central = Buffer.alloc(46), end = Buffer.alloc(22);
  const flags = descriptor ? 8 : 0, crc = crc32(data);
  local.writeUInt32LE(0x04034b50); local.writeUInt16LE(20, 4); local.writeUInt16LE(flags, 6); local.writeUInt16LE(method, 8);
  if (!descriptor) { local.writeUInt32LE(crc, 14); local.writeUInt32LE(packed.length, 18); local.writeUInt32LE(data.length, 22); }
  local.writeUInt16LE(nameBytes.length, 26);
  const tail = Buffer.alloc(descriptor ? 16 : 0);
  if (descriptor) { tail.writeUInt32LE(0x08074b50); tail.writeUInt32LE(crc,4); tail.writeUInt32LE(packed.length,8); tail.writeUInt32LE(data.length,12); }
  central.writeUInt32LE(0x02014b50); central.writeUInt16LE(20,4); central.writeUInt16LE(20,6);
  central.writeUInt16LE(flags,8); central.writeUInt16LE(method,10); central.writeUInt32LE(crc,16);
  central.writeUInt32LE(packed.length,20); central.writeUInt32LE(data.length,24); central.writeUInt16LE(nameBytes.length,28);
  const offset = local.length + nameBytes.length + packed.length + tail.length;
  end.writeUInt32LE(0x06054b50); end.writeUInt16LE(1,8); end.writeUInt16LE(1,10);
  end.writeUInt32LE(central.length + nameBytes.length,12); end.writeUInt32LE(offset,16);
  return Buffer.concat([local,nameBytes,packed,tail,central,nameBytes,end]);
}
function parts(bytes) { const end = bytes.length - 22; return {end, directory: bytes.readUInt32LE(end + 16)}; }
function declaration() {
  return {kind: 'cpb_v15_confirmation_data_selection_v1', status: 'prospective_selection_before_download_or_numeric_access', ...GEOMETRY,
    resumption: {path: '.temp/fixture-admission.json',bytes: 2,sha256: 'a'.repeat(64)},
    sources: SYMBOLS.map(symbol => ({symbol,
      zip_url: `https://data.binance.vision/data/spot/monthly/klines/${symbol}/1m/${symbol}-1m-2024-02.zip`,
      checksum_url: `https://data.binance.vision/data/spot/monthly/klines/${symbol}/1m/${symbol}-1m-2024-02.zip.CHECKSUM`})),
    metadata_sources: METADATA.map(row => ({...row})),
    confirmation: {role: 'confirmation_only', training_use: false, head_fit: false, normalizer_fit: false, probes: 32, history_legal: 256, encoder_history:128},
    license: {publisher:'Binance Vision',commercial_use_authorized:false,dataset_and_derived_artifacts_published:false,attribution_required:true,
      data_license:'CC BY-NC-SA 4.0 plus Binance Vision Dataset Terms',terms_version:'1.0',terms_updated:'2026-08-26'}};
}
test('artificial ZIP stored and deflated members preserve exact CSV bytes', () => {
  for (const method of [0,8]) for (const descriptor of [false,true]) assert.deepEqual(extractCSV(zip(payload,{method,descriptor}),member),payload);
  assert.equal(crc32(Buffer.from('123456789')), 0xcbf43926);
});
test('publisher checksum requires one exact basename and whole hash spelling', () => {
  const digest = 'A'.repeat(64), name = 'BTCUSDT-1m-2024-02.zip';
  assert.equal(checksumDigest(Buffer.from(`${digest}  ${name}\n`),name),'a'.repeat(64));
  assert.equal(checksumDigest(Buffer.from(`${digest} *${name}\r\n`),name),'a'.repeat(64));
  for (const text of [`${digest}  other.zip`,`${digest}  ../${name}`,`${digest}  ${name}\n${digest}  ${name}\n`,`${digest.slice(1)}  ${name}`,`${digest} ${name}`])
    assert.throws(() => checksumDigest(Buffer.from(text),name));
});
test('ZIP paths, wrong member names and multiple members are refused', () => {
  for (const name of ['../'+member,'nested/'+member,'ETHUSDT-1m-2024-02.csv',member+'\0']) assert.throws(() => extractCSV(zip(payload,{name}),member));
  const multiple = zip(); const {end} = parts(multiple); multiple.writeUInt16LE(2,end+8); multiple.writeUInt16LE(2,end+10);
  assert.throws(() => extractCSV(multiple,member),/one-member/);
});
test('ZIP CRC and declared CSV expansion limits are checked before publication', () => {
  assert.throws(() => extractCSV(zip(),member,10));
  const corrupt = zip(payload,{method:0}), {directory} = parts(corrupt);
  corrupt[30 + Buffer.byteLength(member)] ^= 1;
  assert.throws(() => extractCSV(corrupt,member),/CRC/);
  const oversized = zip(); const p=parts(oversized); oversized.writeUInt32LE(50_000_001,p.directory+24);
  assert.throws(() => extractCSV(oversized,member),/size/);
  assert.ok(directory>30);
});
test('ZIP encryption, symlinks, directories and unknown compression are refused', () => {
  for (const mutate of [
    (bytes,p)=>bytes.writeUInt16LE(1,p.directory+8),
    (bytes,p)=>bytes.writeUInt32LE(0xa0000000,p.directory+38),
    (bytes,p)=>bytes.writeUInt32LE(16,p.directory+38),
    (bytes,p)=>bytes.writeUInt16LE(12,p.directory+10),
  ]) { const bytes=zip(); mutate(bytes,parts(bytes)); assert.throws(() => extractCSV(bytes,member)); }
});
test('ZIP local identity, offset, trailing bytes and data descriptor conflicts are refused', () => {
  for (const mutate of [
    (bytes)=>bytes.writeUInt32LE(1,14),
    (bytes,p)=>bytes.writeUInt32LE(1,p.directory+42),
    (bytes)=>bytes.writeUInt16LE(0,26),
  ]) { const bytes=zip(); mutate(bytes,parts(bytes)); assert.throws(() => extractCSV(bytes,member)); }
  assert.throws(() => extractCSV(Buffer.concat([zip(),Buffer.from('unbound')]),member));
  const descriptor=zip(payload,{descriptor:true}), p=parts(descriptor); descriptor.writeUInt32LE(1,p.directory-12);
  assert.throws(() => extractCSV(descriptor,member),/descriptor/);
});
test('February declaration fixes source order, month, chronology and confirmation-only fitting policy', () => {
  assert.equal(validateDeclaration(declaration()).month,'2024-02');
  for (const alter of [
    value=>value.month='2024-03',value=>value.rows=40320,value=>value.sources.reverse(),
    value=>value.sources[0].zip_url=value.sources[0].zip_url.replace('https:','http:'),
    value=>value.confirmation.training_use=true,value=>value.confirmation.head_fit=true,
    value=>value.confirmation.probes=64,value=>value.license.commercial_use_authorized=true,
    value=>value.license.data_license='MIT',value=>value.metadata_sources.pop(),
  ]) { const value=declaration(); alter(value); assert.throws(()=>validateDeclaration(value)); }
});
test('public preparation APIs refuse a nonmanaged root before any network request', async () => {
  let calls=0; const fetch=async()=>{calls++;throw new Error('unexpected network');};
  for (const run of [prepareMetadata,importConfirmation]) await assert.rejects(run({root:'/tmp/fixture',output:'/tmp/new-data',fetch}),/managed Linux/);
  assert.equal(calls,0);
});
test('missing, failed or incompatible January parity proof is refused before February import', () => {
  const pin = name => ({path:name,bytes:2,sha256:'a'.repeat(64)});
  const value = {kind:'cpb_v15_target_precision_parity_v1',passed:true,status:'passed',train_rows:112,validation_rows:32,
    target_precision:'legacy_f32_then_f64',sources:[pin('source.cpp')],january_caches:['BTC','ETH','SOL'].map(name=>pin(name+'.cache.bin')),
    original_targets:['train','validation'].map(name=>pin(name+'-targets.pt')),all_target_values_bitwise_equal:true,all_masks_and_ids_exact:true,
    February_values_opened:false,encoder_calls:0,head_fits:0,source_review:pin('review.json'),protocol_review:pin('protocol.json'),
    binary:pin('/opt/mityane-geometrics/embeddings/build/fixture/target_prepare'),numeric_result:pin('result.json'),log:pin('log.txt')};
  assert.equal(validateParityReceipt(value),value);
  for(const alter of [row=>row.passed=false,row=>row.status='failed',row=>row.train_rows=111,
    row=>row.target_precision='direct_f64',row=>row.original_targets.pop(),row=>row.sources=[],row=>delete row.binary,
    row=>row.February_values_opened=true,row=>row.all_target_values_bitwise_equal=false,row=>row.log.sha256='x',
    row=>row.january_caches[1]=row.january_caches[0]]) {
    const bad=structuredClone(value);alter(bad);assert.throws(()=>validateParityReceipt(bad));
  }
});
