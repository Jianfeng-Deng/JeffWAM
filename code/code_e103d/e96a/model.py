"""E96a P1: row-major current-only residual projection; E94a M1 unchanged."""
import torch
from torch import nn
from e86.model import AttentionPool, Config
from e94a.model import WorldModel as M1WorldModel
from e90.model import adapter_pass, SOURCES

class SpatialCurrent(AttentionPool):
    def __init__(self, original):
        # Reuse the original modules/parameters, preserving names and values.
        nn.Module.__init__(self)
        self.proj=original.proj
        self.query=original.query
        self.attn=original.attn
        # Creating zero weights must not shift M1's subsequent training RNG.
        with torch.random.fork_rng(devices=[]):
            self.bypass=nn.Linear(self.proj.in_features,self.proj.out_features,bias=False)
            nn.init.zeros_(self.bypass.weight)

    def forward(self, raw):
        if raw.shape[-2]!=8:
            raise ValueError('P1 current requires eight row-major cells per camera')
        return super().forward(raw)+self.bypass(raw)

class WorldModel(M1WorldModel):
    def __init__(self,c=None,recipe='M1',readout_hidden=1024):
        super().__init__(c,recipe=recipe,readout_hidden=readout_hidden)
        self.current=SpatialCurrent(self.current)
