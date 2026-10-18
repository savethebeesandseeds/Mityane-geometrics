// SPDX-License-Identifier: AGPL-3.0-only
import {readFile, access, readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {resolve, dirname} from 'node:path';
import {validateManifest, validateHero, validateProject, flattenMilestones} from '../src/model.js';

export const dashboardDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export async function sourceFingerprint() {
  const files = ['manifest.json', 'package.json'];
  async function collect(folder) {
    for (const entry of await readdir(resolve(dashboardDirectory, folder), {withFileTypes: true})) {
      const file = `${folder}/${entry.name}`;
      if (entry.isDirectory()) await collect(file);
      else if (entry.isFile()) files.push(file);
      else throw new Error(`Dashboard inputs must be ordinary files: ${file}`);
    }
  }
  for (const folder of ['src', 'scripts', 'heroes', 'schemas', 'assets', 'tests']) await collect(folder);
  const hash = createHash('sha256');
  for (const file of files.sort()) hash.update(file).update('\0').update(await readFile(resolve(dashboardDirectory, file))).update('\0');
  return hash.digest('hex');
}
export async function readProject(directory = dashboardDirectory) {
  const json = async file => JSON.parse((await readFile(resolve(directory, file), 'utf8')).replace(/^\uFEFF/, ''));
  const manifest = validateManifest(await json('manifest.json'));
  const heroes = await Promise.all(manifest.heroes.map(async entry => validateHero(await json(entry.file))));
  validateProject(manifest, heroes);
  const root = resolve(directory, '../..');
  for (const {milestone} of flattenMilestones(heroes)) for (const evidence of milestone.evidence || []) {
    await access(resolve(root, evidence.path.split('#')[0]));
  }
  return {manifest, heroes};
}
