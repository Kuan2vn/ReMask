import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]))
from bundle import safe_members,extract
from colab_setup import find_repo,pin_revision
from dynamics import export,panel_summary,persistent_agreement
from remask.data.taxonomy import decode_mask,UIIS_COLOR_TO_COMMON_ID


class ColabAndDynamics(unittest.TestCase):
    def test_extraction_preserves_data_prefix_and_hf_suim_prefix(self):
        for prefix,dest_prefix in [('data',''),('SUIM','data')]:
            with tempfile.TemporaryDirectory() as tmp:
                content=io.BytesIO()
                with zipfile.ZipFile(content,'w') as z:z.writestr(prefix+'/train_val/images/a.jpg',b'image')
                with zipfile.ZipFile(content) as z:extract(z,Path(tmp)/dest_prefix,safe_members(z,prefix))
                self.assertEqual((Path(tmp)/dest_prefix/prefix/'train_val/images/a.jpg').read_bytes(),b'image')

    def test_rejects_unsafe_and_duplicate_members(self):
        for names in [('data/../outside',),('/data/file',),('data\\file',),('data/C:/file',),
                      ('other/file',),('data/file','data/file')]:
            content=io.BytesIO()
            with zipfile.ZipFile(content,'w') as z:
                for name in names:z.writestr(name,b'x')
            with zipfile.ZipFile(content) as z:
                with self.assertRaises(ValueError):safe_members(z)

    def test_existing_destination_symlink_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'repo';root.mkdir()
            outside=Path(tmp)/'outside';outside.mkdir()
            (root/'data').symlink_to(outside,target_is_directory=True)
            content=io.BytesIO()
            with zipfile.ZipFile(content,'w') as z:z.writestr('data/file',b'x')
            with zipfile.ZipFile(content) as z:
                with self.assertRaises(ValueError):extract(z,root,safe_members(z))
            self.assertFalse((outside/'file').exists())

    def test_manual_and_nested_repo_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'SND_ReMask/ReMask'
            (root/'experiments/rebuttal').mkdir(parents=True)
            (root/'.git').mkdir();(root/'experiments/rebuttal/run.py').touch()
            self.assertEqual(find_repo(root,content=tmp),root.resolve())
            self.assertEqual(find_repo(tmp,content=tmp),root.resolve())
            self.assertEqual(find_repo(tmp,hint=root,content='/nonexistent'),root.resolve())

    def test_drive_pin_real_newline_and_resume_ignores_new_remote_head(self):
        def git(root,*args):
            return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.DEVNULL).strip()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);origin=root/'origin';origin.mkdir()
            git(origin,'init','-b','main')
            (origin/'file').write_text('one')
            git(origin,'add','file');git(origin,'-c','user.name=Test','-c','user.email=test@local','commit','-m','one')
            checkout=root/'checkout'
            subprocess.check_call(['git','clone',str(origin),str(checkout)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            output=root/'drive';first=pin_revision(checkout,output)
            self.assertEqual((output/'CODE_REVISION.txt').read_bytes(),(first+'\n').encode())
            (origin/'file').write_text('two');git(origin,'add','file')
            git(origin,'-c','user.name=Test','-c','user.email=test@local','commit','-m','two')
            self.assertEqual(pin_revision(checkout,output),first)
            (output/'CODE_REVISION.txt').write_text(first+'\\n')
            self.assertEqual(pin_revision(checkout,output),first)

    def test_actual_notebook_setup_cell_reuses_manual_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);origin=root/'origin';origin.mkdir()
            def git(*args):
                subprocess.check_call(['git','-C',str(origin),*args],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            git('init','-b','main')
            (origin/'experiments/rebuttal').mkdir(parents=True)
            (origin/'experiments/rebuttal/run.py').write_text('# test checkout')
            git('add','.');git('-c','user.name=Test','-c','user.email=test@local','commit','-m','fixture')
            repo=root/'SND_ReMask/ReMask'
            subprocess.check_call(['git','clone',str(origin),str(repo)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            nb=json.loads((HERE/'colab.ipynb').read_text(encoding='utf-8'))
            source=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
            output=root/'drive_output'
            source=source.replace("'/content/drive/MyDrive/ReMask_component_experiments_v2'",repr(str(output)))
            mounted=[];google=SimpleNamespace(drive=SimpleNamespace(mount=mounted.append))
            original=Path.cwd();namespace={}
            try:
                os.chdir(repo)
                with patch.dict(sys.modules,{'google.colab':google}):exec(compile(source,'colab_setup_cell','exec'),namespace)
            finally:os.chdir(original)
            self.assertEqual(namespace['REPO'],repo.resolve())
            self.assertEqual(mounted,['/content/drive'])
            self.assertEqual((output/'CODE_REVISION.txt').read_bytes(),(namespace['commit']+'\n').encode())

    def test_uiis_archive_rgb_and_hf_raw_indices_decode_identically(self):
        colors=list(UIIS_COLOR_TO_COMMON_ID)
        rgb=Image.fromarray(np.array([colors],dtype=np.uint8))
        indexed=Image.fromarray(np.arange(8,dtype=np.uint8)[None])
        expected=np.array([[0,6,5,2,3,1,4,7]],dtype=np.uint8)
        np.testing.assert_array_equal(decode_mask(rgb,'uiis_auto',8),expected)
        np.testing.assert_array_equal(decode_mask(indexed,'uiis_auto',8),expected)
        np.testing.assert_array_equal(decode_mask(indexed.convert('P'),'uiis_auto',8),expected)

    def test_all_notebook_python_cells_parse_and_run_flags_are_shared(self):
        notebook=json.loads((HERE/'colab.ipynb').read_text(encoding='utf-8'))
        sources=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
        for source in sources:ast.parse(source)
        smoke=next(s for s in sources if 'COMMON =' in s)
        self.assertIn("'--save-pixel-diagnostics'",smoke)
        self.assertIn("*COMMON, '--session-hours'",smoke)
        self.assertIn('ReMask_component_experiments_v2',sources[0])
        self.assertFalse(any(c.get('outputs') for c in notebook['cells']))

    def test_stability_requires_all_agreements_and_unchanged_label(self):
        teachers=[np.array([[[0,1,2,3]]],dtype=np.uint8),np.array([[[0,1,3,3]]],dtype=np.uint8),np.array([[[0,1,3,3]]],dtype=np.uint8)]
        mismatches=[np.array([[[False,True,False,False]]]),np.zeros((1,1,4),bool),np.zeros((1,1,4),bool)]
        np.testing.assert_array_equal(persistent_agreement(list(zip(teachers,mismatches))),[[[True,False,False,True]]])

    def test_absent_class_is_missing_and_background_does_not_hide_fg(self):
        panel=dict(predicted_class_support=[99,1,0,0,0,0,0,0],predicted_class_mismatch=[0,1,0,0,0,0,0,0],
                   mismatches=1,pixels=100,raw_kl=.1,teacher_drift=None)
        row=panel_summary(panel)
        self.assertEqual(row['mismatch_percent'],1)
        self.assertEqual(row['foreground_mismatch_percent'],100)
        self.assertEqual(row['macro_class_mismatch_percent'],50)
        self.assertIsNone(row['fish_mismatch_percent'])

    def test_report_uses_committed_maps_and_excludes_reset_from_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'F/A';(folder/'diagnostics/maps').mkdir(parents=True)
            rows=[]
            for step in range(5):
                relative=f'diagnostics/maps/step_{step:06d}.npz'
                np.savez_compressed(folder/relative,teacher=np.zeros((1,2,2),np.uint8),
                                    mismatch=np.zeros((1,2,2),bool),names=np.array(['fixed.jpg']))
                panel=dict(images=['fixed.jpg'],predicted_class_support=[4,0,0,0,0,0,0,0],
                    predicted_class_mismatch=[0]*8,pixels=4,mismatches=0,raw_kl=0.,teacher_drift=None,pixel_map=relative)
                rows.append(dict(local_updates=step,global_updates=step*500,phase='A',unlabeled_same_view_panel=panel))
            (folder/'metrics.json').write_text(json.dumps(rows))
            (folder/'CURRENT.json').write_text(json.dumps(dict(iteration=3)))
            (folder/'PHASE_CONFIG.json').write_text(json.dumps(dict(protocol='uiis_to_suim',diagnostic=True)))
            exported=export(folder,Path(tmp)/'report')
            self.assertEqual(len(exported),4)
            self.assertIsNone(exported[2]['persistent_agreement_percent'])
            self.assertEqual(exported[3]['persistent_agreement_percent'],100.)
            self.assertEqual(exported[3]['stability_window_start'],500)
            self.assertTrue((Path(tmp)/'report/dynamics.png').is_file())
            report=(Path(tmp)/'report/SUPERVISOR_REPORT.md').read_text()
            self.assertIn('uiis_to_suim / F/A [SMOKE: not a method result]',report)


if __name__=='__main__':unittest.main()
