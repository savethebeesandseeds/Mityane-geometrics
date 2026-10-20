import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, renameSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { FILES, identity } from '../../operations/forecasting/direct_embedding_control/versions/v1/source_identity.mjs';

const scratch = '/opt/mityane-geometrics/maintenance-tests';
mkdirSync(scratch, { recursive: true });

function fixture(t) {
  const root = mkdtempSync(`${scratch}/identity-`);
  t.after(() => rmSync(root, { recursive: true, force: true }));
  for (const [index, name] of FILES.entries()) {
    mkdirSync(path.dirname(path.join(root, name)), { recursive: true });
    writeFileSync(path.join(root, name), `fixture ${index}: π\r\n\0`);
  }
  return root;
}

test('source identity preserves the Python ordered length-prefixed UTF-8 golden', t => {
  const root = fixture(t);
  // Generated once with the unchanged source_identity.py, including CRLF,
  // a multibyte character and a NUL in each implementation source.
  assert.equal(identity(root), 'a2a5b14f31164fd052900af56ecce2aa4cf916795f6bf258e354ce0de1ad5467');
  writeFileSync(path.join(root, FILES[0]), 'changed bytes');
  assert.notEqual(identity(root), 'a2a5b14f31164fd052900af56ecce2aa4cf916795f6bf258e354ce0de1ad5467');
});

test('source identity rejects missing sources and directories', t => {
  const root = fixture(t);
  const first = path.join(root, FILES[0]);
  rmSync(first);
  assert.throws(() => identity(root), /ENOENT/);
  mkdirSync(first);
  assert.throws(() => identity(root), /Not an implementation source/);
});

test('source identity resolves repository roots and rejects escaping symlinks', t => {
  const root = fixture(t);
  const alias = `${root}-alias`;
  symlinkSync(root, alias);
  t.after(() => rmSync(alias));
  assert.equal(identity(alias), identity(root));
  const first = path.join(root, FILES[0]);
  const outside = `${root}-outside`;
  renameSync(first, outside);
  t.after(() => rmSync(outside));
  symlinkSync(outside, first);
  assert.throws(() => identity(root), /escapes repository/);
});
