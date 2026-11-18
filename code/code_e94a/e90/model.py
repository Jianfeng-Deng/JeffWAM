import torch
from torch import nn
from e88.model import WorldModel as BaseWorldModel
from e86.model import Config

SOURCES = ('piper', 'yam', 'abc')

class WorldModel(BaseWorldModel):
    def __init__(self, c=None):
        super().__init__(c)
        self.source_embedding = nn.Embedding(len(SOURCES), self.c.width)

    def observation(self, b, training=None):
        o, valid, history = super().observation(b, training)
        source = b.get('source_id', torch.zeros(len(o), dtype=torch.long, device=o.device))
        token = self.source_embedding(source)[:, None] + self.kind[0]
        return torch.cat((o, token), 1), torch.cat((valid, torch.ones(len(o), 1, dtype=torch.bool, device=o.device)), 1), history

def adapter_pass(nmse, cosine):
    return bool(nmse <= .1 and cosine >= .95)
