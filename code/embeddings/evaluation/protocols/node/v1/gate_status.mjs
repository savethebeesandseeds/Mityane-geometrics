// Status inspection only; scientific gate computation stays independently tested.
import {readJSON,requireCondition as check} from '../../../../../shared/artifacts/node/v1/files.mjs';
export function status(value){check(value&&value.gate&&typeof value.gate.passed==='boolean','typed gate status required');return value.gate.passed?0:3;}
if(process.argv[1]?.endsWith('/gate_status.mjs')){
  check(process.argv.length===3,'one exact summary path required');process.exitCode=status(readJSON(process.argv[2]));
}
