// Copyright (c) 2026 Waajacu. See LICENSE.
// Prepared compatibility import only. V16 has one authoritative report.mjs CLI,
// output report.json and source/admission generation; there is no successor run.
import path from 'node:path';
import {fileURLToPath} from 'node:url';
export * from './report.mjs';
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  process.stderr.write('Invoke report.mjs --inputs CARD; this compatibility import is not an executable reporting generation.\n');
  process.exitCode=1;
}
