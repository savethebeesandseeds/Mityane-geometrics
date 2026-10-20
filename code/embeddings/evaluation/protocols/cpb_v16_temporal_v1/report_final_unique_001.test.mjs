// Copyright (c) 2026 Waajacu. See LICENSE.
// Artificial import identity only; no study inputs or model execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as authoritative from './report.mjs';
import * as compatibility from './report_final_unique_001.mjs';
test('prepared compatibility path exports the sole authoritative V16 reporter',()=>{
  for(const name of ['makeReport','pairedScreen','aggregateMarket','aggregateTempo','historicalRetention','planFinalVerification','verifyFinalBarrier'])assert.equal(compatibility[name],authoritative[name],name);
});
