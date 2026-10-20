// Explicit CLI adapter for new capsules. No historical receipt is rewritten.
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {parseArgs} from 'node:util';
import {readJSON,jsonBytes,requireCondition as check} from '../../../../../shared/artifacts/node/v1/files.mjs';
const scopeTargets={configurable_branch_v1:'print-configurable-branch-sources',
  configurable_capacity_v1:'print-configurable-capacity-sources',
  configurable_learned_relation_v1:'print-configurable-learned-relation-sources',
  configurable_content_relation_v1:'print-configurable-content-relation-sources'};
const modes={'verify-admission':'verifyAdmission','bind-quality':'bindQuality','admit-third':'admitThird'};
export async function main(family,createCapsule,argv=process.argv.slice(2)) {
  check(process.platform==='linux'&&fs.existsSync('/.dockerenv')&&process.cwd()==='/mityane','use managed development container');
  const {values,positionals}=parseArgs({args:argv,allowPositionals:true,strict:true,options:{
    build:{type:'string'},original:{type:'string'},branch:{type:'string'},previous:{type:'string'},'source-list':{type:'string'}}});
  check(positionals.length===2,'capsule mode and absolute capsule path required');const [mode,root]=positionals;
  check(['freeze','finish','verify-admission','bind-quality','admit-third'].includes(mode),'metadata-only capsule mode required; tensor bind remains in independent Python audit');
  check(path.isAbsolute(root)&&path.normalize(root)===root,'absolute exact capsule path');
  const repositoryRoot='/mityane';
  const receipt=mode==='freeze'?null:readJSON(path.join(root,'freeze/receipt.json'));
  const mapping=receipt?.migration_implementation?.historical_path_mapping??[];
  const mapped=fragment=>mapping.find(row=>row.recorded_path.includes(fragment))?.current_path;
  const buildDirectory=values.build??receipt?.build_directory;
  const options={repositoryRoot,buildDirectory,originalCapsule:values.original??mapped('rpb-v20-capacity'),
    branchCapsule:values.branch??mapped('rpb-v20-branches'),previousCapsule:values.previous??mapped('rpb-v21-learned-relation')};
  if(mode==='freeze') {
    check(typeof buildDirectory==='string','explicit named build directory required');
    options.gitHead=execFileSync('git',['rev-parse','HEAD'],{cwd:repositoryRoot,encoding:'utf8'}).trim();
    options.sourceList=values['source-list']?fs.readFileSync(values['source-list'],'utf8'):
      execFileSync('make',['--no-print-directory','-s',scopeTargets[family],`BUILD_DIR=${buildDirectory}`],{cwd:repositoryRoot,encoding:'utf8'});
    options.commandOutput=(binary,args)=>{
      check(path.dirname(binary)===buildDirectory&&args.length===1&&args[0]==='--source-id','only declared binary source identity query');
      return execFileSync(binary,args,{cwd:repositoryRoot,encoding:'utf8'});
    };
  }
  const capsule=createCapsule(options),method=modes[mode]??mode;
  check(typeof capsule[method]==='function','supported capsule metadata mode');
  const result=capsule[method](root)??{protocol:family,status:'passed',mode};
  process.stdout.write(jsonBytes(result,{sortKeys:false,indent:0}));return result;
}
