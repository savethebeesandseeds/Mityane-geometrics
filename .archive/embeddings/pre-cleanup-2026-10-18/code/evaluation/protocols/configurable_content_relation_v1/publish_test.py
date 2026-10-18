#!/usr/bin/env python3
"""Artificial publisher integrity mutations; no quality tensors or encoder work."""
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import types
import unittest
import publish as p

ROOT=Path('/embedding')
spec=importlib.util.spec_from_file_location('relation_publication_arithmetic',
    ROOT/'code/evaluation/protocols/configurable_branch_v1/publish.py')
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def tensor(values,dtype='LongStorage',shape=None):
    return dict(dtype=dtype,shape=shape if shape is not None else [len(values)],values=list(values))


def checkpoint(point=512):
    return dict(
        artifact_kind='rpb_content_ordered_relation_inference_checkpoint_v1',model_tag='RPB-v22',
        architecture_id='configurable-fixed-spectral-plus-zero-start-content-ordered-lag-v1',
        native_layout='learned-base-shape-plus-zero-start-content-ordered-lag-projection;'
            'unchanged-fixed-four-per-semantic-pair;'
            'lag-bank-odd-even-no-additive-support;canonical-pair-feature-i-feature-j-lag;'
            'original-tick-lags1-2-4-8;four-endpoint-common-support;'
            'joint-endpoint-centering;min-common2;energy-floor1e-12',
        dtype='float32',training_device='cuda:0',
        training_updates=tensor([point],shape=[]),
        config_integer_v1=tensor([3,128,9,8,64,32,3,4,1,1,2,0,0,256,128]),
        config_number_v1=tensor([0.,1e-5,.25,1.,1e-6,1.],'DoubleStorage'),
        channel_ids=tensor([0,1,2]),relation_lags_v1=tensor([1,2,4,8]),
        relation_minimum_common=tensor([2],shape=[]),
        relation_energy_floor=tensor([1e-12],'DoubleStorage',[]))


def pairs(masters=(941101,941102)):
    return [dict(master=master,id=name,repetition=rep,view=view)
            for master in masters for name in p.PAIR_IDS for rep in p.REPETITIONS for view in p.gate.VIEWS]


class PublicationTest(unittest.TestCase):
    def setUp(self):
        p.READS.clear()

    def tearDown(self):
        p.READS.clear()

    def test_allocated_stage_bindings_pass(self):
        p.validate_stage('screen',[941101,941102])
        p.validate_stage('third',[941103])

    def test_stage_cannot_relabel_or_reorder_cohorts(self):
        for stage,masters in [('screen',[941103]),('third',[941101,941102]),
                              ('screen',[941102,941101]),('screen',[941101,941102,941103]),
                              ('confirmation',[941104])]:
            with self.subTest(stage=stage,masters=masters):
                with self.assertRaises(ValueError):
                    p.validate_stage(stage,masters)

    def test_exact_complete_paired_comparison_population(self):
        p.validate_pairs(pairs(),[941101,941102])
        p.validate_pairs(pairs((941103,)),[941103])

    def test_undeclared_valid_method_comparison_rejected(self):
        rows=pairs()
        # Both methods are legitimate saved methods, but this is not one of
        # the card's four comparisons and must not substitute for one.
        rows[0]['id']='candidate_native_512_minus_learned_20_0'
        with self.assertRaises(ValueError):
            p.validate_pairs(rows,[941101,941102])

    def test_missing_or_duplicate_pairs_rejected(self):
        for rows in (pairs()[:-1],pairs()+[copy.deepcopy(pairs()[0])]):
            with self.assertRaises(ValueError):
                p.validate_pairs(rows,[941101,941102])

    def test_comparison_role_mutations_rejected(self):
        for key,value in [('master',941103),('view','training'),('repetition','rep-0000')]:
            rows=pairs();rows[0][key]=value
            with self.assertRaises(ValueError):
                p.validate_pairs(rows,[941101,941102])

    def test_identically_named_candidate_and_control_cannot_overwrite(self):
        candidate=p.prediction_key(941101,'learned_20_512','rep-2701','validation_intact',True)
        control=p.prediction_key(941101,'learned_20_512','rep-2701','validation_intact',False)
        self.assertNotEqual(candidate,control)
        predictions={candidate:'new candidate decisions',control:'preserved control decisions'}
        self.assertEqual(len(predictions),2)
        self.assertEqual(candidate,(941101,'candidate_learned_20_512','rep-2701','validation_intact'))
        self.assertEqual(control,(941101,'learned_20_512','rep-2701','validation_intact'))

    def test_prediction_identity_keeps_cohort_repetition_and_view(self):
        keys={p.prediction_key(master,'learned_20_512',rep,view,new)
              for master in (941101,941102) for rep in p.REPETITIONS for view in p.gate.VIEWS
              for new in (False,True)}
        self.assertEqual(len(keys),24)

    def test_imported_query_reads_are_retained(self):
        with tempfile.TemporaryDirectory(prefix='relation-publish-reads-') as directory:
            path=Path(directory)/'artificial-query.txt';path.write_text('saved query fixture')
            digest=p.sha(path)
            p.merge_query_reads({path:digest})
            self.assertEqual(p.READS,{path:digest})
            p.merge_query_reads({path:digest})

    def test_changed_query_bytes_rejected(self):
        with tempfile.TemporaryDirectory(prefix='relation-publish-change-') as directory:
            path=Path(directory)/'artificial-query.txt';path.write_text('initial bytes')
            digest=p.remember(path)
            path.write_text('changed bytes')
            with self.assertRaises(ValueError):
                p.merge_query_reads({path:digest})

    def test_wrong_imported_digest_rejected(self):
        with tempfile.TemporaryDirectory(prefix='relation-publish-digest-') as directory:
            path=Path(directory)/'artificial-query.txt';path.write_text('correct bytes')
            with self.assertRaises(ValueError):
                p.merge_query_reads({path:'0'*64})

    def test_linked_input_roles_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='relation-publish-links-') as directory:
            root=Path(directory);target=root/'original.txt';target.write_text('retained fixture')
            alias=root/'symlink.txt';alias.symlink_to(target)
            with self.assertRaises(ValueError):
                p.remember(alias)
            hardlink=root/'hardlink.txt';os.link(target,hardlink)
            with self.assertRaises(ValueError):
                p.remember(target)

    def test_exact_candidate_checkpoint_metadata_passes(self):
        codec=types.SimpleNamespace(text=lambda value:value)
        p.validate_checkpoint_metadata(checkpoint(),codec,a,512)
        p.validate_checkpoint_metadata(checkpoint(0),codec,a,0)

    def test_wrong_checkpoint_text_contracts_rejected(self):
        codec=types.SimpleNamespace(text=lambda value:value)
        for key,value in [('artifact_kind','rpb_configurable_inference_checkpoint_v1'),
                          ('artifact_kind','rpb_learned_ordered_relation_inference_checkpoint_v1'),
                          ('model_tag','RPB-v20'),('architecture_id','another-architecture'),
                          ('model_tag','RPB-v21'),
                          ('architecture_id','configurable-fixed-spectral-plus-zero-start-learned-ordered-lag-v1'),
                          ('native_layout','reversed-coordinate-layout'),('dtype','float64'),
                          ('training_device','cpu')]:
            cp=checkpoint();cp[key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    p.validate_checkpoint_metadata(cp,codec,a,512)

    def test_wrong_checkpoint_tensor_contracts_rejected(self):
        codec=types.SimpleNamespace(text=lambda value:value)
        mutations=[('training_updates',tensor([0],shape=[])),
                   ('config_integer_v1',tensor([3,256,9,8,64,32,3,4,1,1,2,0,0,256,128])),
                   ('channel_ids',tensor([2,1,0])),('relation_lags_v1',tensor([1,2,4,16])),
                   ('relation_minimum_common',tensor([1],shape=[])),
                   ('relation_energy_floor',tensor([1e-6],'DoubleStorage',[])),
                   ('relation_energy_floor',tensor([float('nan')],'DoubleStorage',[])),
                   ('relation_energy_floor',tensor([1e-12],'FloatStorage',[])),
                   ('channel_ids',tensor([0,1,2],shape=[1,3]))]
        for key,value in mutations:
            cp=checkpoint();cp[key]=value
            with self.subTest(key=key,value=value):
                with self.assertRaises(ValueError):
                    p.validate_checkpoint_metadata(cp,codec,a,512)

    def test_exclusive_result_publication_retains_existing_bytes(self):
        with tempfile.TemporaryDirectory(prefix='relation-publish-output-') as directory:
            path=Path(directory)/'summary.json'
            p.write_new(path,dict(retained=True))
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):
                p.write_new(path,dict(retained=False))
            self.assertEqual(path.read_bytes(),before)

    def test_nonfinite_or_missing_aggregate_rejected(self):
        for values in ([],[None],[float('nan')],[float('inf')]):
            with self.assertRaises(ValueError):
                p.average(values)


if __name__=='__main__':
    unittest.main(verbosity=2)
