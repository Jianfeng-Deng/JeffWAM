import random,os
from pathlib import Path
import numpy as np,torch
from .common import sha

def code_hash():
    import hashlib
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()
def save(path,model,optimizer,step,sampler,refs,extra=None):
    path=Path(path)
    if path.exists():raise FileExistsError('Never overwrite checkpoint: '+str(path))
    device=next(model.parameters()).device
    d=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,sampler=sampler.state_dict(),refs=refs,config=vars(model.c),rng=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),extra=extra or {},code_sha256=code_hash())
    if device.type=='cuda':d['cuda_rng']=torch.cuda.get_rng_state(device)
    torch.save(d,path)
def load(path,model,optimizer,sampler,refs):
    d=torch.load(path,map_location='cpu',weights_only=False)
    if d.get('code_sha256')!=code_hash():raise ValueError('Checkpoint code hash mismatch')
    if d['refs']!=refs or d['config']!=vars(model.c):raise ValueError('Checkpoint data/code/config contract mismatch')
    model.load_state_dict(d['model']);optimizer.load_state_dict(d['optimizer']);sampler.load_state_dict(d['sampler']);torch.set_rng_state(d['rng']);np.random.set_state(d['numpy']);random.setstate(d['python'])
    if next(model.parameters()).device.type=='cuda':torch.cuda.set_rng_state(d['cuda_rng'],next(model.parameters()).device)
    return d['step']
