import os, random
from pathlib import Path
import numpy as np
import torch
from .common import source_hash, object_hash

def save(path, model, optimizer, scheduler, sampler, step, config, manifest_hash, extra=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    d = dict(schema='e90-checkpoint-v1', model=model.state_dict(), optimizer=optimizer.state_dict(),
             scheduler=scheduler.state_dict(), sampler=sampler.state_dict(), step=step,
             config=config, config_hash=object_hash(config), manifest_hash=manifest_hash,
             code_hash=source_hash(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
             torch_rng=torch.get_rng_state(), extra=extra or {})
    if next(model.parameters()).is_cuda:
        d['cuda_rng'] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix('.tmp')
    torch.save(d, tmp)
    os.rename(tmp, path)

def restore(path, model, optimizer, scheduler, sampler, config, manifest_hash):
    d = torch.load(path, map_location='cpu', weights_only=False)
    if d['code_hash'] != source_hash() or d['config_hash'] != object_hash(config) or d['manifest_hash'] != manifest_hash:
        raise ValueError('Resume code/config/data contract changed')
    model.load_state_dict(d['model']); optimizer.load_state_dict(d['optimizer'])
    scheduler.load_state_dict(d['scheduler']); sampler.load_state_dict(d['sampler'])
    random.setstate(d['python_rng']); np.random.set_state(d['numpy_rng']); torch.set_rng_state(d['torch_rng'])
    if next(model.parameters()).is_cuda:
        torch.cuda.set_rng_state_all(d['cuda_rng'])
    return d['step'], d['extra']

def initialize(path, model):
    """Weights-only continuation for a NEW phase/run; never pretends to be exact resume."""
    d = torch.load(path, map_location='cpu', weights_only=False)
    if d.get('schema') != 'e90-checkpoint-v1':
        raise ValueError('Expected E90 checkpoint')
    model.load_state_dict(d['model'], strict=True)
    return {k: d[k] for k in ('step', 'config_hash', 'manifest_hash', 'code_hash')}
