// SPDX-License-Identifier: AGPL-3.0-only
import {summarize} from '../src/model.js';
import {publishFingerprint, readProject, readPublishedProject, sourceFingerprint} from './project.mjs';
import {updateStandalone} from './build-standalone.mjs';
const audit = process.argv.includes('--audit');
const check = process.argv.includes('--check');
const fingerprint = await (audit ? sourceFingerprint() : publishFingerprint());
const record = audit ? await readProject() : await readPublishedProject();
const {heroes} = record;
const counts = summarize(heroes), goals = heroes.reduce((total, hero) => total + hero.goals.length, 0);
const evidence = audit ? ' All evidence file targets exist.' : ' Published from the Hero JSON.';
console.log(`Validated ${heroes.length} Heroes, ${goals} Goals, ${counts.total} Milestones (${counts.done} done, ${counts.active} in progress).${evidence}`);
await updateStandalone(record, {check, fingerprint, audit});
