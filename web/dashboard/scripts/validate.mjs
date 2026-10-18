// SPDX-License-Identifier: AGPL-3.0-only
import {summarize} from '../src/model.js';
import {readProject, sourceFingerprint} from './project.mjs';
import {updateStandalone} from './build-standalone.mjs';
const fingerprint = await sourceFingerprint();
const record = await readProject(), {heroes} = record;
const counts = summarize(heroes), goals = heroes.reduce((total, hero) => total + hero.goals.length, 0);
console.log(`Validated ${heroes.length} Heroes, ${goals} Goals, ${counts.total} Milestones (${counts.done} done, ${counts.active} in progress). All evidence file targets exist.`);
await updateStandalone(record, {check: process.argv.includes('--check'), fingerprint});
