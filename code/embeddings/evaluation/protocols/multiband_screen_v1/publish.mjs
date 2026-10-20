// Keep the historical refusal: no argument parsing, reads or mutations occur.
import {fileURLToPath} from 'node:url';
import path from 'node:path';
export const refusal = "Retired historical publisher: original source is in .archive/embeddings/pre-cleanup-2026-10-18/code/evaluation/protocols/multiband_screen_v1/publish.py. Historical registries are immutable; use a separately versioned publication protocol for new work.";
export function main(){process.stderr.write(refusal+'\n');process.exitCode=1;return 1;}
if(process.argv[1]&&fileURLToPath(import.meta.url)===path.resolve(process.argv[1]))main();
