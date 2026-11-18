import numpy as np
def normalize(x,eps=1e-8):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),eps)
def parent_ci(values,parents,seed=20261112,n_boot=2000):
    values=np.asarray(values,dtype=float);parents=np.asarray(parents);keep=np.isfinite(values);values=values[keep];parents=parents[keep]
    unique=np.unique(parents)
    if not len(unique):return {'estimate':None,'ci95':None,'parents':0,'samples':0}
    means=np.array([values[parents==p].mean() for p in unique]);rng=np.random.default_rng(seed)
    boots=means[rng.integers(len(means),size=(n_boot,len(means)))].mean(-1)
    return {'estimate':float(means.mean()),'micro_estimate':float(values.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),'parents':len(unique),'samples':len(values),'weighting':'equal parent, equal observations within parent'}
def paired_ci(a,b,parents,**kwargs):return parent_ci(np.asarray(a)-np.asarray(b),parents,**kwargs)
def direction_metrics(logits,labels,parents,n_dirs=8):
    logits=np.asarray(logits);labels=np.asarray(labels,dtype=int);pred=logits.argmax(-1);top3=np.argsort(logits,axis=-1)[:,-3:]
    direction_pred=pred//3 if logits.shape[-1]==24 else pred
    direction_true=labels//3 if logits.shape[-1]==24 else labels
    angle=np.minimum((direction_pred-direction_true)%n_dirs,(direction_true-direction_pred)%n_dirs)*360/n_dirs
    conf=np.zeros((logits.shape[-1],logits.shape[-1]),dtype=int);np.add.at(conf,(labels,pred),1)
    result={'top1':parent_ci(pred==labels,parents),'top3':parent_ci((top3==labels[:,None]).any(-1),parents),'direction_top1':parent_ci(direction_pred==direction_true,parents),'direction_angle_error_deg':parent_ci(angle,parents),'confusion':conf.tolist()}
    if logits.shape[-1]==24:result['amplitude_top1']=parent_ci(pred%3==labels%3,parents)
    return result
def localization(score,mask,valid=None):
    score=np.asarray(score,dtype=float);mask=np.asarray(mask,dtype=bool)
    valid=np.ones(mask.shape,dtype=bool) if valid is None else np.asarray(valid,dtype=bool)
    assert score.shape==mask.shape==valid.shape
    if not (mask&valid).any() or not ((~mask)&valid).any():return None
    scores=score[valid];labels=mask[valid];best=np.flatnonzero(scores==scores.max())
    # Expected hit for ties, rather than favoring the first raster coordinate.
    point=float(labels[best].mean())
    # Exact Mann-Whitney AUROC with half credit for ties.
    order=np.argsort(scores,kind='stable');sorted_scores=scores[order];ranks=np.empty(len(scores),dtype=float)
    edges=np.r_[0,np.flatnonzero(np.diff(sorted_scores))+1,len(scores)]
    ranks[order]=np.repeat((edges[:-1]+1+edges[1:])/2,np.diff(edges))
    npos=int(labels.sum());nneg=len(labels)-npos
    auc=(ranks[labels].sum()-npos*(npos+1)/2)/(npos*nneg)
    return {'pointing':point,'auroc':float(auc),'uniform_pointing_chance':float(npos/len(labels)),'inside_pixels':npos,'outside_pixels':nneg}
def planar_follow(predicted,requested,threshold=.001):
    predicted=np.asarray(predicted);requested=np.asarray(requested)
    pbin=np.floor(((np.arctan2(predicted[...,1],predicted[...,0])+np.pi/8)%(2*np.pi))/(np.pi/4)).astype(int)
    rbin=np.floor(((np.arctan2(requested[...,1],requested[...,0])+np.pi/8)%(2*np.pi))/(np.pi/4)).astype(int)
    return (pbin==rbin)&(np.linalg.norm(predicted,axis=-1)>threshold)
def counterfactual_metrics(predicted,requested,true,parents,threshold=1e-4):
    predicted=np.asarray(predicted);requested=np.asarray(requested);true=np.asarray(true)
    good=np.linalg.norm(predicted,axis=-1)>threshold
    cosine=(normalize(predicted)*normalize(requested)).sum(-1)
    truecos=(normalize(predicted)*normalize(true)).sum(-1)
    follow=planar_follow(predicted,requested,threshold)
    return {'follow_rate':parent_ci(follow,parents),'prediction_instruction_cosine':parent_ci(np.where(good,cosine,np.nan),parents),'prediction_true_cosine':parent_ci(np.where(good,truecos,np.nan),parents),'near_zero_rate':parent_ci(~good,parents),'norm':parent_ci(np.linalg.norm(predicted,axis=-1),parents)}
