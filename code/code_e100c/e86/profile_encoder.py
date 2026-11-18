import time,json
from .resource import Guard
from .common import *
def main():
    guard=Guard('encoder_throughput_probe',120)
    try:
        import torch
        from .vision import FrozenEncoder
        torch.set_num_threads(2);e=FrozenEncoder('cuda:0');results=[]
        for n in (16,64,128,256):
            guard.check();x=torch.rand(n,3,224,224,device='cuda');v=torch.ones(n,224,224,device='cuda');e.images(x,v);torch.cuda.synchronize();t=time.monotonic()
            for _ in range(3):e.images(x,v)
            torch.cuda.synchronize();sec=(time.monotonic()-t)/3;results.append(dict(batch=n,seconds=sec,images_per_s=n/sec,peak_mib=torch.cuda.max_memory_allocated()/2**20))
        atomic(C/'artifacts/encoder_throughput_v1.json',dict(results=results,attention=str(e.model.vision_model.config._attn_implementation)));print(json.dumps(results),flush=True)
    finally:guard.finish()
if __name__=='__main__':main()
