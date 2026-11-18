"""Default action_only; state availability is an actual wall-clock dependency."""
import time,torch
from .inference import predict
from .vision import preprocess

def wait_for_state(available_at_monotonic):
    wait=max(0.,float(available_at_monotonic)-time.monotonic())
    if wait>1:raise ValueError('State availability >1s; reject stale/invalid request')
    if wait:time.sleep(wait)
    return wait
@torch.no_grad()
def infer_images(model,encoder,b,rgb_images,available_at_monotonic,mode='action_only'):
    wait=wait_for_state(available_at_monotonic)
    pp=[preprocess(x) for x in rgb_images];pixels=torch.stack([x[0] for x in pp]);valid=torch.stack([x[1] for x in pp]);features=encoder.images(pixels,valid)
    obs=b['obs'].clone();obs[:,4]=features.reshape(1,3,8,768);condition=dict(b,obs=obs)
    action=predict(model,condition,mode=mode)
    return action,dict(state_wait_s=wait)
