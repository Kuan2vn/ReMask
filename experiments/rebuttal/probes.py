"""Public-code UWRG plus a label-free magnitude-matched ColorJitter control."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from util import atomic_json, read_json


def generic_probe(images, amplitude):
    result=[]
    for image in images:
        for op in torch.randperm(4).tolist():
            magnitude = min(.49, .2*amplitude) if op==3 else .5*amplitude
            value = (float(torch.rand(()))*2-1)*magnitude
            function=(TF.adjust_brightness,TF.adjust_contrast,TF.adjust_saturation,TF.adjust_hue)[op]
            image=function(image, value if op==3 else 1+value)
        result.append(image.clamp(0,1))
    return torch.stack(result)


def calibration(path, data, signature, strength, underwater, smoke=False):
    path=Path(path)
    if path.exists():
        record=read_json(path)
        if record['dataset_signature']!=signature:
            raise ValueError('Calibration belongs to a different dataset/strength')
        if record['underwater_strength']==strength:
            return record
    files=sorted((Path(data)/'train_val/images').glob('*.jpg'))
    rng=np.random.default_rng(1)
    selected=sorted(rng.choice(len(files),min(4 if smoke else 32,len(files)),replace=False).tolist())
    images=torch.stack([TF.to_tensor(Image.open(files[i]).convert('RGB').resize((256,128),Image.Resampling.BICUBIC)) for i in selected])
    def distortion(function, value):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1729)
            return float(torch.stack([(function(images,value)-images).abs().mean() for _ in range(3)]).mean())
    target=distortion(underwater,strength)
    low,high=0.,1.5
    for _ in range(8):
        middle=(low+high)/2
        if distortion(generic_probe,middle)<target:
            low=middle
        else:
            high=middle
    amplitude=(low+high)/2
    result=dict(dataset_signature=signature, underwater_strength=strength, generic_amplitude=amplitude,
                underwater_rgb_mad=target, generic_rgb_mad=distortion(generic_probe,amplitude),
                calibration_resolution=[256,128], draws=3, seed=1729,
                images=[files[i].name for i in selected], manual_labels_used=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,result)
    return result


def checkpoint_residual_blocks():
    from core.models.resnet import Bottleneck
    from torch.utils.checkpoint import checkpoint
    original=Bottleneck.forward
    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint(original,self,x,use_reentrant=False,preserve_rng_state=True)
        return original(self,x)
    Bottleneck.forward=forward


def install_probe(trainer, phase, fitted_amplitude):
    original=trainer._underwater_reliability_perturb
    if phase['probe']=='generic':
        maximum=max(phase['strength'],1e-8)
        def generic_with_isolated_rng(x,strength):
            devices=[x.device] if x.is_cuda else []
            step=int(getattr(getattr(trainer,'EXPERIMENT',None),'seen_step',0))
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(phase.get('seed',1))*1000003+int(phase.get('offset',0))+step)
                return generic_probe(x,fitted_amplitude*strength/maximum)
        trainer._underwater_reliability_perturb=generic_with_isolated_rng
    elif phase['probe']=='identity':
        trainer._underwater_reliability_perturb=lambda x,strength:x
    if phase['aggregation']=='mean':
        # Retain the author's component/stride/min-size code and change only its reduction.
        import types
        mean_globals=dict(trainer._segmentwise_median_stability.__globals__)
        from scipy import ndimage
        class MeanRegions:
            def __getattr__(self,name):
                return ndimage.mean if name=='median' else getattr(ndimage,name)
        mean_globals['ndimage']=MeanRegions()
        trainer._segmentwise_median_stability=types.FunctionType(
            trainer._segmentwise_median_stability.__code__,mean_globals)
    warmup=max(1,phase['warmup_steps'])
    trainer._uwrg_probe_strength=lambda iteration,max_iters,max_strength,warmup_ratio:max_strength*min(1.,(iteration+1)/warmup)
    return original
