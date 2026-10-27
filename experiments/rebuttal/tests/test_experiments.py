"""Checks for contrasts, ranking ties, gradient parity, GT exclusion and resume integrity."""
import copy
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE/'native'))
sys.path.append(str(REPO/'baselines/underwater'))
from recipes import ARMS,CORE,PROTOCOLS,ordered_arms,phases
from metrics import quality,overlap,boundary_counts
from util import WARM_BANKS,atomic_json,bank_hash,copy_banks,digest,read_json,seed_warm_banks,verified_pointer,warm_bank_root
from probes import checkpoint_residual_blocks,generic_probe,install_probe
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

    def test_warm_bank_archives_provenance_and_cache_reverification(self):
        import importlib.util
        spec_mod=importlib.util.spec_from_file_location('ub_assets',REPO/'baselines/underwater/underwater_baselines/assets.py')
        ub_assets=importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(ub_assets)
        SOURCES=ub_assets.SOURCES
        manifest=read_json(HERE/'assets/warm_banks/WARM_BANKS.json')
        bundle=read_json(HERE/'dataset_bundle.json')
        self.assertEqual(manifest,WARM_BANKS)
        for protocol in PROTOCOLS:
            spec=WARM_BANKS[protocol]
            self.assertEqual(spec['source_sha256'],SOURCES[protocol]['sha256'],protocol)
            self.assertEqual(spec['dataset_bundle_sha256'],bundle['sha256'],protocol)
            archive=HERE/spec['archive']
            self.assertTrue(archive.is_file(),protocol)
            self.assertEqual(digest(archive),spec['archive_sha256'],protocol)
            unpacked=warm_bank_root(protocol)
            self.assertEqual(bank_hash(unpacked),spec['banks_sha256'],protocol)
            self.assertEqual(digest(unpacked/'CTR.p'),spec['ctr_p_sha256'],protocol)
            self.assertEqual(digest(unpacked/'CTR_O.p'),spec['ctr_o_p_sha256'],protocol)
            with self.assertRaisesRegex(ValueError,'source SHA mismatch'):
                seed_warm_banks(protocol,unpacked,'0'*64)
            # Tamper with a file inside the unpacked cache and verify warm_bank_root self-heals.
            sample_png=next((unpacked/'CTR').glob('*.png'))
            original_bytes=sample_png.read_bytes()
            sample_png.write_bytes(b'corrupted-mask-bytes')
            self.assertNotEqual(bank_hash(unpacked),spec['banks_sha256'])
            repaired=warm_bank_root(protocol)
            self.assertEqual(bank_hash(repaired),spec['banks_sha256'],protocol)
            self.assertEqual((repaired/'CTR'/sample_png.name).read_bytes(),original_bytes)

    def test_local_bank_io_boundary_and_checked_hash_have_no_scope_error(self):
        import control as control_module
        from control import Control
        import local_bank_io
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path=Path(tmp)
            cache_dir=tmp_path/'cache'
            out_dir=tmp_path/'phase_out'
            out_dir.mkdir(parents=True)
            old_env=os.environ.get('REMASK_BANK_CACHE')
            orig_init=Control.__init__
            orig_copy=control_module.copy_banks
            orig_hash=control_module.bank_hash
            orig_ens=trainer._ensure_remask_candidates
            orig_cand=trainer.run_candidate
            orig_gen=trainer.gen_lb_info
            orig_build=trainer._build_target_loader
            try:
                os.environ['REMASK_BANK_CACHE']=str(cache_dir)
                working=local_bank_io.prepare_phase(out_dir,fresh=True)
                for folder in ('CTR','CTR_O'):
                    (working/folder).mkdir(parents=True,exist_ok=True)
                    (working/folder/'sample.png').write_bytes(b'png-bytes')
                for idx_name in ('CTR.p','CTR_O.p'):
                    (working/idx_name).write_bytes(b'pickle-bytes')
                expected_hash=orig_hash(working)
                local_bank_io.install_worker(Control,control_module)
                task=dict(phase=phases('F',PROTOCOLS[0],smoke=True)[0],output=str(out_dir),device='cpu',
                          seconds=60.,smoke=True,signature=dict(test=True),
                          source=dict(sha256=WARM_BANKS[PROTOCOLS[0]]['source_sha256']),
                          activation_checkpointing=False,evaluation_size=[32,32],panel_images=2,quality_stride=2)
                ctrl=Control(task,trainer)
                ctrl.segments_init=ctrl.resources.setdefault('segments',[dict(start_local_updates=0)])
                ckpt=out_dir/'model_000001.pth'
                ckpt.write_bytes(b'checkpoint-bytes')
                try:
                    ctrl.boundary(1,str(ckpt),{})
                except control_module.SessionPaused:
                    pass
                pointer=read_json(out_dir/'CURRENT.json')
                self.assertEqual(pointer['banks_sha256'],expected_hash)
                verified_cache=cache_dir/'verified'/expected_hash
                self.assertTrue(verified_cache.is_dir())
                self.assertEqual(orig_hash(verified_cache),expected_hash)
            finally:
                if old_env is None:
                    os.environ.pop('REMASK_BANK_CACHE',None)
                else:
                    os.environ['REMASK_BANK_CACHE']=old_env
                Control.__init__=orig_init
                control_module.copy_banks=orig_copy
                control_module.bank_hash=orig_hash
                trainer._ensure_remask_candidates=orig_ens
                trainer.run_candidate=orig_cand
                trainer.gen_lb_info=orig_gen
                trainer._build_target_loader=orig_build

    def test_probe_rng_isolation_preserves_main_training_rng_stream(self):
        x=torch.rand(2,3,16,16)
        orig_perturb=trainer._underwater_reliability_perturb
        orig_strength=trainer._uwrg_probe_strength
        try:
            for probe_name in ('underwater','generic','identity'):
                p=phases('F' if probe_name=='underwater' else 'G-generic-med' if probe_name=='generic' else 'G-confidence',PROTOCOLS[0])[0]
                install_probe(trainer,p,0.25)
                raw_probe=trainer._underwater_reliability_perturb
                seen_step=7
                def isolated_probe(images,strength,_raw=raw_probe):
                    devices=[images.device] if images.is_cuda else []
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(1*1000003+p['offset']+seen_step)
                        return _raw(images,strength)
                torch.manual_seed(42)
                before_state=torch.get_rng_state()
                expected_next=torch.rand(5)
                torch.set_rng_state(before_state)
                out1=isolated_probe(x,0.2)
                after_probe_draw=torch.rand(5)
                torch.testing.assert_close(expected_next,after_probe_draw,rtol=0,atol=0)
                out2=isolated_probe(x,0.2)
                torch.testing.assert_close(out1,out2,rtol=0,atol=0)
        finally:
            trainer._underwater_reliability_perturb=orig_perturb
            trainer._uwrg_probe_strength=orig_strength

    def test_warm_bank_target_loader_and_training_step_integration(self):
        from core.configs import cfg
        from core.datasets.dataset_path_catalog import DatasetCatalog
        for protocol in PROTOCOLS:
            unpacked=warm_bank_root(protocol)
            for folder,idx_name in (('CTR','CTR.p'),('CTR_O','CTR_O.p')):
                pngs={p.name for p in (unpacked/folder).glob('*.png')}
                self.assertEqual(len(pngs),1525,(protocol,folder))
                l2f,f2l=pickle.loads((unpacked/idx_name).read_bytes())
                self.assertTrue(set(f2l.keys()).issubset(pngs),(protocol,idx_name))
                self.assertGreater(len(f2l),1400,(protocol,idx_name))
                self.assertTrue(set(l2f.keys()).issubset(set(range(8))),(protocol,idx_name))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            unpacked=warm_bank_root(PROTOCOLS[0])
            sample_names=sorted(p.stem for p in (unpacked/'CTR').glob('*.png'))[:8]
            view=root/'SUIM/orin_data'
            images_dir=view/'train_val/images'
            images_dir.mkdir(parents=True)
            for stem in sample_names:
                with Image.open(unpacked/'CTR'/f'{stem}.png') as mask_img:
                    w,h=mask_img.size
                Image.new('RGB',(w,h),(60,110,150)).save(images_dir/f'{stem}.jpg')
            out_dir=root/'phase_out'
            out_dir.mkdir(parents=True)
            seed_warm_banks(PROTOCOLS[0],out_dir,WARM_BANKS[PROTOCOLS[0]]['source_sha256'],view=view,smoke=True)
            c=cfg.clone()
            c.defrost()
            c.OUTPUT_DIR=str(out_dir)
            c.DATASETS.TARGET_TRAIN='suim_train'
            c.MODEL.NUM_CLASSES=8
            c.TCR.OPEN=True
            c.INPUT.SOURCE_INPUT_SIZE_TRAIN=(64,48)
            c.INPUT.TARGET_INPUT_SIZE_TRAIN=(64,48)
            c.INPUT.INPUT_SIZE_TEST=(64,48)
            c.INPUT.INPUT_SCALES_TRAIN=(1.,1.)
            c.SOLVER.BATCH_SIZE=2
            c.freeze()
            called=[]
            orig_cand=trainer.run_candidate
            previous_dir=DatasetCatalog.DATASET_DIR
            try:
                DatasetCatalog.DATASET_DIR=str(root)
                trainer.run_candidate=lambda *a,**kw:called.append(True)
                trainer._ensure_remask_candidates(c,None,None,'cpu',None)
                self.assertEqual(called,[],'Warm bank should skip initial run_candidate')
                dataset,_=trainer._build_target_loader(c,2,False)
                self.assertEqual(len(dataset.ids),8)
                trans_img,trans_label,name,_,img_full,mix_label=dataset[0]
                self.assertEqual(tuple(trans_img.shape),(3,48,64))
                self.assertEqual(tuple(img_full.shape),(3,48,64))
                self.assertEqual(tuple(trans_label.shape),(48,64))
                self.assertEqual(tuple(mix_label.shape),(48,64))
                valid=((mix_label>=0)&(mix_label<8))|(mix_label==255)
                self.assertTrue(bool(valid.all()))
                conv=torch.nn.Conv2d(3,8,kernel_size=1)
                logits=conv(torch.stack([trans_img,img_full]).float())
                targets=torch.zeros((2,48,64),dtype=torch.long)
                loss=torch.nn.functional.cross_entropy(logits,targets,ignore_index=255)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertIsNotNone(conv.weight.grad)
                self.assertTrue(bool(torch.isfinite(conv.weight.grad).all()))
            finally:
                trainer.run_candidate=orig_cand
                DatasetCatalog.DATASET_DIR=previous_dir


if __name__=='__main__':unittest.main()
