"""Appearance strata fixed from unlabeled images;not physical turbidity annotations."""
import numpy as np
from PIL import Image


def describe(path):
    with Image.open(path) as image:
        rgb=np.asarray(image.convert('RGB').resize((256,128)),dtype='float32')/255
    luma=rgb@np.array([.2126,.7152,.0722])
    return dict(brightness=float(luma.mean()),contrast=float(luma.std()),
                red_blue_ratio=float((rgb[...,0].mean()+.001)/(rgb[...,2].mean()+.001)),
                clipping_fraction=float((rgb.max(2)>.98).mean()))


def thresholds(paths):
    rows=[describe(path) for path in paths]
    return {name:float(np.quantile([r[name] for r in rows],.75 if name=='clipping_fraction' else .25))
            for name in rows[0]}


def strata(values,cutoffs):
    groups=['all']
    for name,group in [('brightness','low_light_proxy'),('contrast','low_contrast_proxy'),('red_blue_ratio','blue_green_proxy')]:
        if values[name]<=cutoffs[name]:groups.append(group)
    if values['clipping_fraction']>max(.01,cutoffs['clipping_fraction']):groups.append('highlight_clipping_proxy')
    return groups
