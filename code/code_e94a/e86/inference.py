import torch
from .common import seed_for
@torch.no_grad()
def predict(model,b,mode='action_only',noise_samples=4,steps=10):
    preds=[]
    for k in range(noise_samples):preds.append(model.infer(b,mode,steps,seed=seed_for(86,'eval',k))[0])
    return torch.stack(preds).mean(0)
