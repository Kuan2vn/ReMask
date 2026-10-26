"""Full confusion/boundary metrics; tie-aware sampled-pixel reliability metrics."""
import cv2
import numpy as np


def confusion(prediction, target):
    valid = (target >= 0) & (target < 8)
    return np.bincount(8*target[valid].astype('int64') + prediction[valid], minlength=64).reshape(8,8)


def overlap(matrix):
    matrix = np.asarray(matrix, dtype='float64')
    tp, actual, predicted = matrix.diagonal(), matrix.sum(1), matrix.sum(0)
    union = actual + predicted - tp
    iou = np.divide(tp, union, out=np.zeros(8), where=union>0)
    recall = np.divide(tp, actual, out=np.zeros(8), where=actual>0)
    dice = np.divide(2*tp, actual+predicted, out=np.zeros(8), where=(actual+predicted)>0)
    return dict(miou=float(iou.mean()), fg_miou=float(iou[1:].mean()), class_iou=iou.tolist(),
                class_accuracy=recall.tolist(), class_dice=dice.tolist(), mean_accuracy=float(recall.mean()),
                pixel_accuracy=float(tp.sum()/max(actual.sum(),1)),
                frequency_weighted_iou=float((actual*iou).sum()/max(actual.sum(),1)), confusion=matrix.astype('int64').tolist())


def boundary(mask, ratio=.02):
    h,w = mask.shape
    dilation = max(1, int(round(ratio*np.sqrt(h*h+w*w))))
    padded = np.pad(mask.astype('uint8'), 1)
    eroded = cv2.erode(padded, np.ones((3,3), 'uint8'), iterations=dilation)[1:h+1,1:w+1]
    return mask.astype(bool) & ~eroded.astype(bool)


def boundary_counts(prediction, target):
    valid = (target >= 0) & (target < 8)
    values = []
    for c in range(8):
        p, y = boundary((prediction==c)&valid), boundary((target==c)&valid)
        values.append(((p&y).sum(), (p|y).sum()))
    return np.asarray(values, dtype='int64')


def quality(score, correct):
    score, errors = np.asarray(score).reshape(-1), 1-np.asarray(correct, dtype='float64').reshape(-1)
    if not len(score):
        return None
    order = np.argsort(-score, kind='stable')
    ranked, errors = score[order], errors[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ranked)!=0)+1]
    counts = np.diff(np.r_[starts,len(ranked)])
    group_errors = np.add.reduceat(errors, starts)
    before = np.cumsum(group_errors)-group_errors
    offsets = np.arange(len(ranked))-np.repeat(starts, counts)+1
    expected_errors = np.repeat(before,counts) + offsets*np.repeat(group_errors/counts,counts)
    risks = expected_errors/np.arange(1,len(ranked)+1)
    result = dict(aurc=float(risks.mean()), error_rate=float(errors.mean()), pixels=len(score))
    for coverage in (.5,.75,.9):
        count = max(1,int(np.ceil(coverage*len(score))))
        result[f'precision_at_{int(coverage*100)}'] = float(1-risks[count-1])
    positives = errors.sum()
    if 0 < positives < len(errors):
        # Error score = -reliability; group-average ranks handle score ties.
        mean_error_ranks = np.repeat((starts+1+starts+counts)/2, counts)
        result['error_auroc'] = float(((errors*mean_error_ranks).sum()-positives*(positives+1)/2)/(positives*(len(errors)-positives)))
    else:
        result['error_auroc'] = None
    return result


def paired_interval(values, seed=1, draws=2000):
    values=np.asarray(values,dtype='float64')
    if not len(values):
        return None
    rng=np.random.default_rng(seed)
    means=np.asarray([values[rng.integers(0,len(values),len(values))].mean() for _ in range(draws)])
    return dict(mean=float(values.mean()), ci95=np.quantile(means,[.025,.975]).tolist(), n_images=len(values))
