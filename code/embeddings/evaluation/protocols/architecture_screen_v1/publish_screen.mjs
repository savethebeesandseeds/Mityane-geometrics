// Metadata summary only. Numerical validators and their frozen arithmetic remain independent.
import {cli,isMain,publish} from '../node/v1/metadata.mjs';
export const run = args => publish('architecture_screen_v1',args);
if(isMain(import.meta.url))cli('architecture_screen_v1','publish');
