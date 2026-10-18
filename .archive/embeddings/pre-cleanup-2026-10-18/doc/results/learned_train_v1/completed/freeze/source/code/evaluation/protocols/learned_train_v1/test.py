#!/usr/bin/env python3
"""Meaningful artificial arithmetic and field-boundary checks, also under -O."""
import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec=importlib.util.spec_from_file_location('saved_train_diagnostic',Path(__file__).with_name('analysis.py'))
d=importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class DiagnosticTests(unittest.TestCase):
    def test_constant_coordinates_have_zero_rank(self):
        stats,cov=d.moments([[3.,-2.],[3.,-2.]])
        self.assertEqual(stats['exact_constant_coordinates'],2)
        self.assertEqual(stats['floor_coordinates'],2)
        self.assertEqual(stats['covariance_participation_rank'],0.)
        self.assertEqual(cov,[[0.,0.],[0.,0.]])

    def test_correlated_coordinates_have_rank_one(self):
        stats,cov=d.moments([[-1.,-2.],[1.,2.]])
        self.assertEqual(cov,[[1.,2.],[2.,4.]])
        self.assertEqual(stats['covariance_participation_rank'],1.)
        self.assertEqual([c['population_std'] for c in stats['coordinates']],[1.,2.])

    def test_nonfinite_and_ragged_train_rows_rejected(self):
        for rows in ([[1.],[float('nan')]],[[1.],[2.,3.]]):
            with self.assertRaises(ValueError):
                d.moments(rows)

    def test_pair_orientation_and_norms(self):
        result=d.pair_metrics([[1.,2.],[4.,6.]],['a','a'],[1,1])
        self.assertEqual(result['pairs'][0]['signed_difference'],[3.,4.])
        self.assertEqual(result['mean_pair_l2'],5.)
        reversed_result=d.pair_metrics([[4.,6.],[1.,2.]],['a','a'],[1,1])
        self.assertEqual(reversed_result['pairs'][0]['signed_difference'],[-3.,-4.])
        self.assertEqual(reversed_result['mean_pair_l2'],5.)

    def test_pair_identity_and_support_mutations_rejected(self):
        for ids,valid in [(['a','b'],[1,1]),(['a','a'],[1,0]),(['a','a','a','a'],[1,1,1,1])]:
            with self.assertRaises(ValueError):
                d.check_pairs(ids,valid)

    def test_training_delta_and_pair_delta_are_distinct(self):
        result=d.change_metrics([[0.,0.],[3.,4.]],[[1.,2.],[4.,6.]],['a','a'],[1,1])
        self.assertEqual(result['changed_cells'],4)
        self.assertEqual(result['signed_coordinate_mean_change'],[1.,2.])
        self.assertEqual(result['pair_difference_changes']['pairs'][0]['signed_difference'],[0.,0.])
        self.assertEqual(result['pair_difference_changes']['exact_zero_pairs'],1)

    def test_decoder_covariance_cancellation_is_retained(self):
        stats,cov=d.moments([[-1.,-1.],[1.,1.]])
        weights=[[0.]*64 for unused in range(128)]
        weights[0][0]=1.;weights[0][1]=-1.
        result=d.decoder_metrics(weights,stats['coordinates'],cov,0,2)
        self.assertAlmostEqual(result['std_adjusted_block_rss'],math.sqrt(2))
        self.assertEqual(result['covariance_centered_linear_contribution_rms'],0.)

    def test_decoder_cross_covariance_is_counted(self):
        stats,cov=d.moments([[-1.,-1.],[1.,1.]])
        weights=[[0.]*64 for unused in range(128)]
        weights[0][0]=1.;weights[0][1]=1.
        result=d.decoder_metrics(weights,stats['coordinates'],cov,0,2)
        self.assertEqual(result['covariance_centered_linear_contribution_rms'],2.)

    def test_validation_payload_and_head_weights_are_rejected_before_read(self):
        a=d.Archive.__new__(d.Archive)
        a.allow={'training_values'}
        with patch.object(zipfile.ZipFile,'read',side_effect=RuntimeError('should not read storage')):
            for key in ('validation_values','deleted_values','ridge_weights','labels_scoring_only'):
                with self.assertRaises(ValueError):
                    a.tensor(key)

    def test_selected_storage_stride_and_offset(self):
        class Node: pass
        with tempfile.TemporaryDirectory(prefix='learned-train-engineering-') as directory:
            path=Path(directory)/'artificial.pt'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('fixture/data/0',struct.pack('<6d',99.,1.,2.,3.,4.,88.))
                archive.writestr('fixture/data/unselected',b'invalid numeric bytes')
            a=d.Archive.__new__(d.Archive)
            a.path=path;a.allow={'training_values'};a.root=Node()
            a.root.training_values=d.TensorRef(d.StorageRef(path,'fixture/data/0','DoubleStorage','cpu'),1,[2,2],[1,2])
            # A poison validation descriptor exists but its bytes are never read.
            a.root.validation_values=d.TensorRef(d.StorageRef(path,'fixture/data/unselected','DoubleStorage','cpu'),0,[2,2],[2,1])
            original_root=d.ROOT
            d.ROOT=Path(directory)
            try:
                self.assertEqual(a.tensor('training_values','DoubleStorage',[2,2]),[1.,3.,2.,4.])
                with self.assertRaises(ValueError):
                    a.tensor('validation_values')
            finally:
                d.ROOT=original_root

    def test_validation_storage_alias_rejected(self):
        class Node: pass
        a=d.Archive.__new__(d.Archive)
        a.allow={'training_values'};a.root=Node()
        storage=d.StorageRef(Path('/tmp/artificial.pt'),'fixture/shared','DoubleStorage','cpu')
        a.root.training_values=d.TensorRef(storage,0,[2],[1])
        a.root.validation_values=d.TensorRef(storage,2,[2],[1])
        with patch.object(zipfile.ZipFile,'read',side_effect=RuntimeError('alias must reject before byte read')):
            with self.assertRaises(ValueError):
                a.tensor('training_values','DoubleStorage',[2])

    def test_exclusive_publication_preserves_existing_file(self):
        with tempfile.TemporaryDirectory(prefix='learned-train-publish-') as directory:
            path=Path(directory)/'receipt.json'
            d.write_new(path,dict(original=True))
            previous=path.read_bytes()
            with self.assertRaises(FileExistsError):
                d.write_new(path,dict(original=False))
            self.assertEqual(path.read_bytes(),previous)


if __name__=='__main__':
    unittest.main(verbosity=2)
