// Prospective Node metadata implementation; historical Python records stay intact.
import {cli,isMain,prepare} from '../node/v1/metadata.mjs';
export const run = args => prepare('two_component_information_v1',args);
if(isMain(import.meta.url))cli('two_component_information_v1','prepare');
