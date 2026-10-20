// Prospective Node metadata implementation; historical Python records stay intact.
import {cli,isMain,prepare} from '../node/v1/metadata.mjs';
export const run = args => prepare('multiband_screen_v1',args);
if(isMain(import.meta.url))cli('multiband_screen_v1','prepare');
