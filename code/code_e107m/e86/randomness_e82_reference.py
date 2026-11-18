import hashlib, random
import numpy as np
import torch

def derive(seed,*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str,(seed,)+parts)).encode()).digest()[:8],'little')%(2**63-1)

def generator(seed,*parts,device='cpu'):
    return torch.Generator(device=device).manual_seed(derive(seed,*parts))

def seed_all(seed):
    random.seed(seed); np.random.seed(seed%2**32); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

class Streams:
    """Stateless per-stage/update/slot streams: W never advances A or data RNG."""
    def __init__(self,seed,stage,step,slot=0): self.key=(seed,stage,step,slot)
    def gen(self,name,device='cpu'): return generator(*self.key,name,device=device)
    def normal(self,name,shape,device='cpu'):
        return torch.randn(shape,generator=self.gen(name,device),device=device)
    def uniform(self,name,shape,device='cpu'):
        return torch.rand(shape,generator=self.gen(name,device),device=device)
