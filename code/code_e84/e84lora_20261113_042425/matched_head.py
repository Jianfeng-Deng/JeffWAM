import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import sys
sys.dont_write_bytecode=True
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
from common import *
from encoder import Encoder
from evaluation_data import load_features,motion_dataset,candidate_texts
from train_heads import fit_standardize,classify
from metrics import direction_metrics

def main():
 root=P/'matched_frozen';root.mkdir(exist_ok=True);features=load_features('frozen');enc=Encoder('cpu')
 for arm in ['left','right']:
  out=root/f'motion_robot_{arm}_h50.json'
  if out.exists():continue
  df,x,*_=motion_dataset(features,'robot',arm,50);tr=(df.probe_split=='probe-train').to_numpy();va=~tr;xt,xv,mean,std=fit_standardize(x[tr],x[va]);text=enc.text(candidate_texts(arm,'robot','zh'));y=df.label.to_numpy();ps=df.parent_component.to_numpy()[va];scores={};pars=None
  for condition in ['true','shuffled']:
   torch.manual_seed(20261112);rng=np.random.default_rng(20261112);model=nn.Sequential(nn.Linear(6912,25),nn.GELU(),nn.Linear(25,768));pars=sum(p.numel() for p in model.parameters());opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3);xx=torch.tensor(xt);yy=torch.tensor(y[tr] if condition=='true' else np.random.default_rng(20261112).permutation(y[tr]));tv=torch.tensor(text)
   for ep in range(40):
    for ids in np.array_split(rng.permutation(len(xx)),max(1,int(np.ceil(len(xx)/128)))):
     opt.zero_grad(set_to_none=True);log=F.normalize(model(xx[ids]),dim=-1)@tv.T/.07;loss=F.cross_entropy(log,yy[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
   with torch.inference_mode():scores[condition]=np.concatenate([(F.normalize(model(torch.tensor(b)),dim=-1)@tv.T/.07).numpy() for b in np.array_split(xv,max(1,int(np.ceil(len(xv)/256))))])
  np.savez_compressed(root/f'motion_robot_{arm}_h50_predictions.npz',labels=y[va],parents=ps,**scores)
  atomic(out,dict(arm=arm,horizon=50,trainable_parameters=pars,lora_parameters=196608,relative_parameter_difference=pars/196608-1,training_rule='frozen encoder, same40epochs/lr1e-3 as original head, hidden25. 1.94% fewer parameters than LoRA; compute not matched',metrics={k:direction_metrics(v,y[va],ps) for k,v in scores.items()}));print('MATCHED',arm,pars,flush=True)
if __name__=='__main__':main()
