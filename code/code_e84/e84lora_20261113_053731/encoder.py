import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel,AutoTokenizer
from common import *
MODEL='/mnt/workspace/dengjianfeng/models/siglip2'
WORDS=['right arm','left arm','right gripper','left gripper','orange zip tie','locking head','tip']
CONTROLS=['blue zip tie','green zip tie','bicycle wheel','purple cloud','wooden spoon','random unrelated text']
PROCESSOR={'do_convert_rgb':True,'do_resize':True,'size':{'height':224,'width':224},'resample':2,'do_rescale':True,'rescale_factor':1/255,'do_normalize':True,'image_mean':[.5,.5,.5],'image_std':[.5,.5,.5],'spatial_transform':'full FOV aspect preserve resize, edge pad square; no ROI/crop; affine explicitly stored','source':'https://huggingface.co/google/siglip2-base-patch16-224/blob/main/preprocessor_config.json','source_processor_commit':'997aaec','local_weight_revision':'75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2','difference':'official pixel normalization and bilinear kernel; explicit aspect padding replaces official direct square warp'}
def preprocess(rgb):
    im=Image.fromarray(rgb).convert('RGB');w,h=im.size
    scale=min(224/w,224/h);nw,nh=round(w*scale),round(h*scale);left=(224-nw)//2;top=(224-nh)//2
    x=np.array(im.resize((nw,nh),Image.Resampling.BILINEAR));x=np.pad(x,((top,224-nh-top),(left,224-nw-left),(0,0)),mode='edge')
    affine=np.array([[nw/w,0,left],[0,nh/h,top],[0,0,1]],dtype=np.float32)
    return torch.from_numpy(x.copy()).permute(2,0,1).float()/127.5-1,affine,np.array([left,top,left+nw,top+nh],dtype=np.int16)
class Encoder:
    def __init__(self,device='cpu',adapter=None):
        self.device=device;self.model=AutoModel.from_pretrained(MODEL,local_files_only=True).eval().to(device)
        self.model.requires_grad_(False)
        self.tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,use_fast=True)
        if adapter:
            from lora_module import inject
            inject(self.model,layers=4);self.model.load_state_dict(torch.load(adapter,map_location='cpu',weights_only=True),strict=False)
    @torch.inference_mode()
    def text(self,texts,batch=64):
        out=[]
        for i in range(0,len(texts),batch):
            enc=self.tok([s.lower() for s in texts[i:i+batch]],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(self.device)
            z=self.model.get_text_features(**enc)
            if not isinstance(z,torch.Tensor):z=z.pooler_output
            out.append(F.normalize(z.float(),dim=-1).cpu())
        return torch.cat(out).numpy()
    @torch.inference_mode()
    def images(self,pixels):
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.startswith('cuda')):
            z=self.model.vision_model(pixel_values=pixels.to(self.device))
        return z.pooler_output.float().cpu().numpy(),z.last_hidden_state.float().cpu().numpy()
    @torch.inference_mode()
    def patch_vectors(self,patches):
        x=torch.as_tensor(patches,dtype=torch.float32,device=self.device);shape=x.shape
        head=self.model.vision_model.head
        # Singleton attention has exactly one value; avoids 196 duplicated query computations.
        att=head.attention;d=x.shape[-1]
        v=F.linear(x,att.in_proj_weight[2*d:],None if att.in_proj_bias is None else att.in_proj_bias[2*d:])
        v=att.out_proj(v);v=v+head.mlp(head.layernorm(v))
        return F.normalize(v,dim=-1).cpu().numpy()

    @torch.inference_mode()
    def global_images(self,pixels):
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.startswith('cuda')):
            z=self.model.vision_model(pixel_values=pixels.to(self.device))
        return z.pooler_output.float().cpu().numpy(),None
