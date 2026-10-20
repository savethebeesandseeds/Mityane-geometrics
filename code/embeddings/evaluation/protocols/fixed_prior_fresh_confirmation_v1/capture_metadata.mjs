import {cli,isMain,captureMetadata} from '../node/v1/metadata.mjs';
export const run = args => captureMetadata('fixed_prior_fresh_confirmation_v1',args);
if(isMain(import.meta.url))cli('fixed_prior_fresh_confirmation_v1','capture');
