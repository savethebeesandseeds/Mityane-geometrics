#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-only
// Prepare dashboard artwork with Node and the managed image's locked native libraries.
import {createHash} from 'node:crypto';
import {existsSync} from 'node:fs';
import {readFile, writeFile, realpath, stat} from 'node:fs/promises';
import {dirname, extname, isAbsolute, relative, resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
export const brandingDirectory = resolve(root, 'web/dashboard/assets/branding');
export const nativeBinary = '/opt/mityane-geometrics/web/build/prepare-branding-native';
const budget = 512 * 1024;
const digest = data => createHash('sha256').update(data).digest('hex');
const json = async path => JSON.parse((await readFile(path, 'utf8')).replace(/^\uFEFF/, ''));
const within = (parent, path) => {
  const child = relative(parent, path);
  return child !== '..' && !child.startsWith('../') && !isAbsolute(child);
};

export function selectedArtwork(selection, manifest) {
  const ids = manifest.heroes.map(hero => hero.id);
  if (!selection || Object.keys(selection).sort().join(',') !== 'global,heroes' || !selection.heroes ||
      Object.keys(selection.heroes).sort().join('\0') !== [...ids].sort().join('\0'))
    throw new Error('Assign artwork to the global brand and every manifest Hero, with no extra IDs.');
  const selected = [['global', selection.global, 384], ...ids.map(id => [id, selection.heroes[id], 256])];
  if (selected.some(([id, source]) => !/^[a-z0-9-]+$/.test(id) || typeof source !== 'string'))
    throw new Error('Artwork assignments require valid Hero IDs and PNG source paths.');
  if (new Set(selected.map(([, source]) => source)).size !== selected.length)
    throw new Error('Each Hero and the global brand need distinct source artwork.');
  return selected;
}

function native(arguments_) {
  const result = spawnSync(nativeBinary, arguments_, {maxBuffer: budget + 65536});
  if (result.error) throw new Error(`Cannot run artwork helper; run make dashboard-branding-native: ${result.error.message}`);
  if (result.status !== 0) throw new Error(result.stderr.toString().trim() || `Artwork helper failed: ${result.signal || result.status}`);
  return result.stdout;
}

export async function prepareArtwork(selection, manifest) {
  const selected = selectedArtwork(selection, manifest);
  const sources = await realpath(resolve(root, 'web/branding'));
  const encoder = native(['--version']).toString().trim();
  const files = new Map(), records = [];
  for (const [id, sourcePath, size] of selected) {
    const source = await realpath(resolve(sources, sourcePath));
    if (!within(sources, source) || extname(source) !== '.png')
      throw new Error('Source artwork must be a PNG inside web/branding/.');
    const data = native([source, String(size)]), filename = `${id}.webp`;
    files.set(filename, data);
    records.push({id, source: relative(root, source), sourceSHA256: digest(await readFile(source)),
      file: filename, sha256: digest(data), width: size, height: size, bytes: data.length});
  }
  const total = records.reduce((sum, record) => sum + record.bytes, 0);
  if (total > budget) throw new Error(`Prepared artwork exceeds the unchanged 512 KiB budget: ${total} bytes.`);
  const config = {mark: 'global.webp', portrait: 'global.webp', defaultHero: 'global.webp',
    heroes: Object.fromEntries(manifest.heroes.map(({id}) => [id, `${id}.webp`]))};
  const provenance = {source: 'web/branding/dashboard-assignments.json', format: 'WebP RGBA', quality: 82,
    resize: 'GdkPixbuf bilinear', encoder, totalBytes: total, budgetBytes: budget, assets: records};
  files.set('branding.json', Buffer.from(`${JSON.stringify(config, null, 2)}\n`));
  files.set('provenance.json', Buffer.from(`${JSON.stringify(provenance, null, 2)}\n`));
  return {files, total, heroCount: manifest.heroes.length};
}

export async function updateArtwork(files, {check = false, destination = brandingDirectory} = {}) {
  for (const [filename, data] of files) {
    const target = resolve(destination, filename);
    if (await stat(target).then(value => value.isFile(), () => false) && (await readFile(target)).equals(data)) continue;
    if (check) throw new Error(`Prepared branding is out of date: ${filename}. Run refresh.ps1.`);
    await writeFile(target, data);
  }
}

async function main() {
  const arguments_ = process.argv.slice(2);
  if (arguments_.includes('--help') || arguments_.includes('-h')) {
    console.log('Usage: prepare-branding.mjs [--check]\nPrepare dashboard artwork using the managed image\'s locked native libraries.');
    return;
  }
  if (arguments_.some(value => value !== '--check')) throw new Error('Usage: prepare-branding.mjs [--check]');
  if (process.platform !== 'linux' || !existsSync('/.dockerenv') || root !== '/mityane')
    throw new Error('Use container.ps1 and the managed mityane-geometrics container.');
  const {files, total, heroCount} = await prepareArtwork(await json(resolve(root, 'web/branding/dashboard-assignments.json')),
    await json(resolve(root, 'web/dashboard/manifest.json')));
  const check = arguments_.includes('--check');
  await updateArtwork(files, {check});
  console.log(`${check ? 'Verified' : 'Prepared'} global artwork and ${heroCount} unique Hero images: ${total} / ${budget} bytes.`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
