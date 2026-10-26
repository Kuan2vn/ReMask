"""CPU-loaded prototype buffers must follow an already placed classifier."""
import io
from pathlib import Path
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'native'))
import trainer
from core.configs import cfg
from core.models.classifier import ASPP_Classifier_V2


class PrototypeResume(unittest.TestCase):
    def test_cpu_state_allocates_new_buffers_on_classifier_device(self):
        # Meta is a distinct device available in CPU CI; the old allocator fails here.
        classifier=torch.nn.Linear(4,8,device='meta')
        state=dict(global_prototypes=torch.zeros(8,4),global_prototypes_valid=torch.zeros(8,dtype=torch.bool))
        trainer._ensure_classifier_buffers_from_state(classifier,state)
        self.assertEqual(classifier.global_prototypes.device.type,'meta')
        self.assertEqual(classifier.global_prototypes_valid.device.type,'meta')
        self.assertEqual(classifier.global_prototypes_valid.dtype,torch.bool)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable; meta-device regression still runs')
    def test_cuda_roundtrip_resumes_prototype_loss_and_updates(self):
        c=cfg.clone();c.MODEL.NUM_CLASSES=8;c.REMASK.USE_CONTRASTIVE=True;c.REMASK.LAMBDA=.1
        original=ASPP_Classifier_V2(4,[1],[1],8)
        original.register_buffer('global_prototypes',torch.nn.functional.normalize(torch.randn(8,4),dim=1))
        original.register_buffer('global_prototypes_valid',torch.ones(8,dtype=torch.bool))
        stream=io.BytesIO();torch.save(original.state_dict(),stream);stream.seek(0)
        saved=torch.load(stream,map_location='cpu',weights_only=False)
        classifier=ASPP_Classifier_V2(4,[1],[1],8).to('cuda')
        trainer._ensure_classifier_buffers_from_state(classifier,saved)
        trainer._load_module_state(classifier,saved,strict=True)
        for key in ('global_prototypes','global_prototypes_valid'):
            self.assertEqual(getattr(classifier,key).device,next(classifier.parameters()).device)
            torch.testing.assert_close(getattr(classifier,key).cpu(),saved[key])
        prototypes=torch.randn(2,8,4,device='cuda',requires_grad=True)
        valid=torch.ones(2,8,device='cuda',dtype=torch.bool)
        reliability=torch.full((2,8),.5,device='cuda')
        loss=trainer._prototype_contrastive_loss(c,classifier,prototypes,valid,reliability)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prototypes.grad).all()))
        trainer._update_global_prototypes(c,classifier,prototypes,valid)
        self.assertTrue(bool(torch.isfinite(classifier.global_prototypes).all()))


if __name__=='__main__':
    if not torch.cuda.is_available():raise SystemExit('CUDA preflight needs a GPU runtime and CUDA-enabled PyTorch')
    unittest.main()
