// SPDX-License-Identifier: AGPL-3.0-only
// Embed local, replaceable artwork without depending on a sibling project.
import {readFile, realpath} from 'node:fs/promises';
import {resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';

const directory = fileURLToPath(new URL('../assets/branding/', import.meta.url));
const mimeTypes = {'.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'};
export async function readBranding() {
  const config = JSON.parse((await readFile(resolve(directory, 'branding.json'), 'utf8')).replace(/^\uFEFF/, ''));
  if (!config || typeof config !== 'object' || Array.isArray(config)
      || Object.keys(config).sort().join(',') !== 'defaultHero,heroes,mark,portrait'
      || !config.heroes || typeof config.heroes !== 'object' || Array.isArray(config.heroes)
      || Object.keys(config.heroes).some(id => !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id))) {
    throw new Error('Branding needs mark, portrait, defaultHero and a heroes object keyed by Hero ID.');
  }
  const cache = new Map();
  let total = 0;
  async function artwork(file) {
    if (typeof file !== 'string' || !/^[a-zA-Z0-9_-]+\.(svg|png|jpe?g|webp)$/.test(file)) {
      throw new Error('Branding filenames must identify images directly in assets/branding/.');
    }
    if (cache.has(file)) return cache.get(file);
    const path = await realpath(resolve(directory, file));
    if (!path.startsWith((await realpath(directory)) + sep)) throw new Error('Branding must stay in assets/branding/.');
    const bytes = await readFile(path);
    total += bytes.length;
    if (!bytes.length || total > 512 * 1024) throw new Error('Prepared branding must fit the 512 KiB standalone asset budget.');
    const extension = file.slice(file.lastIndexOf('.'));
    if (extension === '.svg') {
      const svg = bytes.toString('utf8');
      if (!svg.includes('<svg') || /<(?:script|foreignObject|style|image|use)\b|\b(?:href|on[a-z]+)\s*=|url\s*\(|<!DOCTYPE|<!ENTITY/i.test(svg)) {
        throw new Error('SVG branding must be self-contained artwork without scripts, links or external content.');
      }
    }
    const uri = `data:${mimeTypes[extension]};base64,${bytes.toString('base64')}`;
    cache.set(file, uri);
    return uri;
  }
  return {
    mark: await artwork(config.mark), portrait: await artwork(config.portrait),
    defaultHero: await artwork(config.defaultHero),
    heroes: Object.fromEntries(await Promise.all(Object.entries(config.heroes).map(async ([id, file]) => [id, await artwork(file)]))),
  };
}
export function brandingScript(branding) {
  const data = JSON.stringify(branding).replace(/[<>&\u2028\u2029]/g,
    character => `\\u${character.charCodeAt(0).toString(16).padStart(4, '0')}`);
  return `\n(() => {\nconst branding = ${data};\nObject.freeze(branding.heroes);\nwindow.MityaneDashboardBranding = Object.freeze(branding);\n})();\n`;
}
