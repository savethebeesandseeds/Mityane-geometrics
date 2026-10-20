import {cli,isMain,captureMetadata} from '../node/v1/metadata.mjs';
export const run = args => captureMetadata('unit_relation_screen_v1',args);
if(isMain(import.meta.url))cli('unit_relation_screen_v1','capture');
