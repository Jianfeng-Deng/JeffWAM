from .common import *
import torch,numpy as np
import torch.nn.functional as F
from e86.vision import preprocess as sig_preprocess,spatial_grid
DINO=Path('/mnt/workspace/dengjianfeng/eyerobot2-top-stereo/third_party/dinov3')
WEIGHT=Path('/mnt/workspace/dengjianfeng/eyerobot2-top-stereo/checkpoints/pretrained/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
MEAN=(.485,.456,.406);STD=(.229,.224,.225)
def preprocess(rgb):
 x,m,a=sig_preprocess(rgb)
 x=(x+1)*.5
 return (x-torch.tensor(MEAN)[:,None,None])/torch.tensor(STD)[:,None,None],m,a
def pool(p,m,grid):
 if grid==(2,4):return spatial_grid(p,m)
 b,n,d=p.shape;x=p.float().transpose(1,2).reshape(b,d,14,14);v=F.avg_pool2d(m[:,None].float(),16,16)
 return (F.adaptive_avg_pool2d(x*v,grid)/F.adaptive_avg_pool2d(v,grid).clamp_min(1e-6)).flatten(2).transpose(1,2)
class Encoder:
 def __init__(self):
  sys.path.insert(0,str(DINO))
  from dinov3.hub.backbones import dinov3_vitb16
  self.model=dinov3_vitb16(pretrained=False)
  self.model.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
  self.model.eval().requires_grad_(False).to('cuda:0')
 @torch.no_grad()
 def patches(self,x):
  with torch.autocast('cuda',dtype=torch.float16):return self.model.forward_features(x.to('cuda:0'))['x_norm_patchtokens']
