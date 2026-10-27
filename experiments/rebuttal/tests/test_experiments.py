"""Checks for contrasts, ranking ties, gradient parity, GT exclusion and resume integrity."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE/'native'))
from recipes import ARMS,CORE,PROTOCOLS,ordered_arms,phases
from metrics import quality,overlap,boundary_counts
from util import WARM_BANKS,atomic_json,bank_hash,copy_banks,digest,verified_pointer,warm_bank_root
from probes import checkpoint_residual_blocks
import trainer
torch.set_num_threads(2)


class Experiments(unittest.TestCase):
    def test_core_budget_and_shared_prefixes(self):
        for protocol in PROTOCOLS:
            total=0
            for arm in CORE:
                for p in phases(arm,protocol):
                    total+=p['steps']-(p['parent_step'] if p['full_resume'] else 0)
                    self.assertGreaterEqual(p['horizon'],p['steps'])
            self.assertEqual(total,205000)
            self.assertEqual(phases('S-20k',protocol)[0]['offset']+phases('S-20k',protocol)[0]['steps'],30000)
            self.assertEqual(phases('S-UA-continuous',protocol)[0]['warmup_steps'],phases('F',protocol)[0]['warmup_steps'])
        self.assertEqual(ordered_arms(arm='S-27.5k'),['F','S-UA-continuous','S-27.5k'])
        self.assertEqual(set(ordered_arms('all')),set(ARMS))

    def test_tc_changes_only_declared_factors(self):
        full=phases('F',PROTOCOLS[0])[1]
        for arm,expected in [('TC-neither',(False,False)),('TC-Tonly',(True,False)),('TC-Conly',(False,True))]:
            p=phases(arm,PROTOCOLS[0])[0]
            self.assertEqual((p['temporal'],p['classes']),expected)
            for key in ('steps','horizon','lr','warmup_steps','floor','exponent','strength','parent_step','contrastive'):
                self.assertEqual(p[key],full[key],(arm,key))
        self.assertFalse(phases('S-UA-restart',PROTOCOLS[0])[0]['full_resume'])
        self.assertTrue(phases('S-UA-continuous',PROTOCOLS[0])[0]['full_resume'])

    def test_quality_perfect_reverse_ties_and_empty_classes(self):
        perfect=quality([.9,.8,.2,.1],[1,1,0,0])
        reverse=quality([.1,.2,.8,.9],[1,1,0,0])
        self.assertEqual(perfect['error_auroc'],1.)
        self.assertEqual(reverse['error_auroc'],0.)
        tied=quality([.5]*4,[1,1,0,0])
        self.assertAlmostEqual(tied['aurc'],.5)
        self.assertAlmostEqual(tied['error_auroc'],.5)
        self.assertIsNone(quality([],[]))
        self.assertIsNone(quality([1,2],[1,1])['error_auroc'])

    def test_identity_probe_is_not_no_gate(self):
        p=torch.softmax(torch.randn(1,8,8,8),1)
        stability,_,_=trainer._compute_underwater_stability(p,p,p.argmax(1),255,'pixel',1,1)
        self.assertTrue(torch.allclose(stability,p.max(1,keepdim=True)[0]))
        self.assertTrue(bool((.8+.2*stability<1).all()))

    def test_residual_activation_checkpoint_keeps_fp32_outputs_and_gradients(self):
        from core.models.resnet import Bottleneck
        from core.models.layers import FrozenBatchNorm2d
        block=Bottleneck(256,64,norm_layer=FrozenBatchNorm2d).train()
        other=copy.deepcopy(block)
        x=torch.randn(2,256,8,8,requires_grad=True)
        y=block(x); y.square().mean().backward()
        original=Bottleneck.forward
        try:
            checkpoint_residual_blocks()
            z=other(x.detach().clone().requires_grad_(True));z.square().mean().backward()
            torch.testing.assert_close(y,z,rtol=0,atol=0)
            for a,b in zip(block.parameters(),other.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
        finally:Bottleneck.forward=original

    def test_training_and_initial_candidate_loaders_never_need_target_masks(self):
        from core.configs import cfg
        from core.datasets.dataset_path_catalog import DatasetCatalog
        from core.datasets.build import build_transform
        c=cfg.clone();c.MODEL.NUM_CLASSES=8;c.TCR.OPEN=False;c.INPUT.INPUT_SIZE_TEST=(128,96)
        c.INPUT.SOURCE_INPUT_SIZE_TRAIN=(96,64);c.INPUT.TARGET_INPUT_SIZE_TRAIN=(96,64)
        c.INPUT.INPUT_SCALES_TRAIN=(1.,1.)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);images=root/'SUIM/orin_data/train_val/images';images.mkdir(parents=True)
            Image.new('RGB',(128,96),(80,120,160)).save(images/'sample.jpg')
            previous=DatasetCatalog.DATASET_DIR
            try:
                DatasetCatalog.DATASET_DIR=str(root)
                for name,mode in [('suim_II_train','III'),('suim_train','train')]:
                    dataset=DatasetCatalog.get(name,mode,8,transform=build_transform(c,'test' if mode=='III' else 'train',False),cfg=c)
                    sample=dataset[0]
                    self.assertTrue(bool((sample[1]==255).all()),name)
            finally:DatasetCatalog.DATASET_DIR=previous

    def test_pointer_rejects_changed_checkpoint_or_bank(self):
        with tempfile.TemporaryDirectory() as folder:
            phase=Path(folder);bank=phase/'banks/0';bank.mkdir(parents=True)
            for name in ('CTR','CTR_O'):
                (bank/name).mkdir();(bank/name/'x.png').write_bytes(b'label')
            for name in ('CTR.p','CTR_O.p'):(bank/name).write_bytes(b'index')
            checkpoint=phase/'step.pth';checkpoint.write_bytes(b'committed')
            atomic_json(phase/'CURRENT.json',dict(checkpoint=checkpoint.name,sha256=digest(checkpoint),banks='banks/0',banks_sha256=bank_hash(bank)))
            verified_pointer(phase)
            (bank/'CTR/x.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'verification'):verified_pointer(phase)

    def test_boundary_and_confusion_identical_prediction(self):
        target=np.zeros((20,30),dtype='int64');target[4:12,5:15]=1
        result=boundary_counts(target,target)
        np.testing.assert_array_equal(result[:,0],result[:,1])
        self.assertEqual(quality([1,1],[1,1])['precision_at_75'],1.)

    def test_best_model_rolls_back_an_uncommitted_evaluation(self):
        from control import Control
        with tempfile.TemporaryDirectory() as folder:
            control=object.__new__(Control);control.out=Path(folder)
            torch.save(dict(iteration=500,value=torch.tensor([1.])),control.out/'best_previous.pth')
            torch.save(dict(iteration=1000,value=torch.tensor([2.])),control.out/'best.pth')
            control.recover_best(500)
            saved=torch.load(control.out/'best.pth',weights_only=False)
            self.assertEqual(saved['iteration'],500)
            self.assertEqual(float(saved['value'][0]),1.)
            with self.assertRaises(ValueError):control.recover_best(2000)

    def test_notebook_is_clean_and_code_cells_compile(self):
        notebook=json.loads((HERE/'colab.ipynb').read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                self.assertFalse(cell.get('outputs'))
                self.assertIsNone(cell['execution_count'])
                compile(''.join(cell['source']),'colab-cell','exec')

    def test_warm_bank_archives_exist_and_match_hashes(self):
        for protocol in PROTOCOLS:
            spec=WARM_BANKS[protocol]
            archive=HERE/spec['archive']
            self.assertTrue(archive.is_file(),protocol)
            self.assertEqual(digest(archive),spec['archive_sha256'],protocol)
            unpacked=warm_bank_root(protocol)
            self.assertEqual(bank_hash(unpacked),spec['banks_sha256'],protocol)


if __name__=='__main__':unittest.main()
