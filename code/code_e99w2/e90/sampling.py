import collections
import torch
from e86.common import generator
from e86.data import CappedParentSampler
from .model import SOURCES

class MultiSourceSampler:
    """Source weights plus hard cumulative window caps; parent equal inside source.

    A source with no cap (normally Piper) is required to keep every prefix feasible.
    Caps do not apply to entries, parents or batches. RNG and counts resume exactly.
    """
    def __init__(self, rows, seed=90, weights=None, caps=None, demo3_cap=.45):
        self.rows = rows; self.seed = seed; self.count = 0
        self.weights = weights or {'piper': 1.}
        self.caps = caps or {}; self.counts = collections.Counter()
        if set(self.weights) - set(SOURCES) or set(self.caps) - set(SOURCES):
            raise ValueError('Unknown source')
        if any(not 0 <= v <= 1 for v in self.caps.values()) or any(v < 0 for v in self.weights.values()):
            raise ValueError('Invalid source probabilities/caps')
        self.ids = {s: [i for i, r in enumerate(rows) if r.get('source', 'piper') == s] for s in SOURCES}
        self.active = [s for s in SOURCES if self.ids[s] and self.weights.get(s, 0) > 0]
        if not self.active or not any(self.caps.get(s, 1.) == 1 for s in self.active):
            raise ValueError('Need an uncapped positive-weight source for hard prefix feasibility')
        self.samplers = {s: CappedParentSampler([rows[i] for i in self.ids[s]], seed=seed+SOURCES.index(s), cap=demo3_cap) for s in self.active}

    def sample(self, n):
        out = []; g = generator(self.seed, self.count, 'source_sampler')
        for _ in range(n):
            eligible = [s for s in self.active if self.counts[s]+1 <= int((self.count+1)*self.caps.get(s, 1.)+1e-9)]
            w = torch.tensor([self.weights[s] for s in eligible], dtype=torch.float64)
            s = eligible[int(torch.multinomial(w, 1, generator=g))]
            j = self.samplers[s].sample(1)[0]; out.append(self.ids[s][j])
            self.count += 1; self.counts[s] += 1
        return out

    def state_dict(self):
        return dict(count=self.count, counts=dict(self.counts), seed=self.seed, weights=self.weights, caps=self.caps,
                    children={s: x.state_dict() for s, x in self.samplers.items()})

    def load_state_dict(self, d):
        if (d['seed'], d['weights'], d['caps']) != (self.seed, self.weights, self.caps):
            raise ValueError('Source sampler config changed')
        self.count = d['count']; self.counts = collections.Counter(d['counts'])
        for s, state in d['children'].items(): self.samplers[s].load_state_dict(state)
