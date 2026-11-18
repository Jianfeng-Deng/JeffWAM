import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np,torch,hashlib,copy
from torch import nn
import torch.nn.functional as F
from heads import AlignmentHead,DirectionHead,DisplacementHead
torch.set_num_threads(2)
def fit_standardize(train,test):
    mean=train.mean(0).astype('f4');std=np.maximum(train.std(0),.01).astype('f4')
    return (train-mean)/std,(test-mean)/std,mean,std
def train_classifier(x,labels,text_features=None,seed=20261112,epochs=40):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=AlignmentHead(x.shape[1],text_features.shape[1]) if text_features is not None else nn.ModuleDict({'direction':DirectionHead(x.shape[1],8),'amplitude':DirectionHead(x.shape[1],3)})
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    x=torch.as_tensor(x,dtype=torch.float32);y=torch.as_tensor(labels,dtype=torch.long);text=None if text_features is None else torch.as_tensor(text_features,dtype=torch.float32)
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/128)))):
            opt.zero_grad(set_to_none=True)
            if text is not None:logits=model(x[ids])@text.T/.07;loss=F.cross_entropy(logits,y[ids])
            else:loss=F.cross_entropy(model['direction'](x[ids]),y[ids]//3)+F.cross_entropy(model['amplitude'](x[ids]),y[ids]%3)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    model.eval();return model
@torch.inference_mode()
def classify(model,x,text_features=None):
    parts=[]
    for batch in np.array_split(x,max(1,int(np.ceil(len(x)/256)))):
        v=torch.as_tensor(batch,dtype=torch.float32)
        if text_features is not None:logits=model(v)@torch.as_tensor(text_features,dtype=torch.float32).T/.07
        else:logits=(model['direction'](v)[:,:,None]+model['amplitude'](v)[:,None,:]).reshape(-1,24)
        parts.append(logits.numpy())
    return np.concatenate(parts)
def fit_displacement(x,state,text,target,use_language=True,seed=20261112,epochs=60):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=DisplacementHead(x.shape[1],state.shape[1],text.shape[1],use_language);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    xx,ss,tt,yy=[torch.as_tensor(a,dtype=torch.float32) for a in [x,state,text,target]]
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(len(x)),max(1,int(np.ceil(len(x)/128)))):
            opt.zero_grad(set_to_none=True);pred=model(xx[ids],ss[ids],tt[ids]);loss=F.mse_loss(pred,yy[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    model.eval();return model
@torch.inference_mode()
def displacement(model,x,state,text):
    out=[]
    for ids in np.array_split(np.arange(len(x)),max(1,int(np.ceil(len(x)/256)))):
        out.append(model(*[torch.as_tensor(a[ids],dtype=torch.float32) for a in [x,state,text]]).numpy())
    return np.concatenate(out)
