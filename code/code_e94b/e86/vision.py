"""Frozen, local-only SigLIP2. No adapters; fixed 224, E84 FOV preprocessing."""
import os,re
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
import numpy as np,torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
MODEL='/mnt/workspace/dengjianfeng/models/siglip2'
PREPROCESS=dict(version='e86_siglip2_e84_fov_grid2x4_v1',resolution=224,rgb=True,resize='aspect_preserve_round_bilinear',padding='edge_center',mean=[.5]*3,std=[.5]*3,crop=None,patch_grid=[14,14],cache_grid=[2,4],cache_dtype='float16',target='fixed_spatial_average_stop_grad',official_difference='E84 aspect padding replaces direct square warp')
def preprocess(rgb):
    im=Image.fromarray(rgb).convert('RGB');w,h=im.size;s=min(224/w,224/h);nw,nh=round(w*s),round(h*s);l=(224-nw)//2;t=(224-nh)//2
    x=np.array(im.resize((nw,nh),Image.Resampling.BILINEAR));x=np.pad(x,((t,224-nh-t),(l,224-nw-l),(0,0)),mode='edge')
    valid=np.zeros((224,224),np.float32);valid[t:t+nh,l:l+nw]=1
    return torch.from_numpy(x.copy()).permute(2,0,1).float()/127.5-1,torch.from_numpy(valid),np.array([[nw/w,0,l],[0,nh/h,t],[0,0,1]],np.float32)
def spatial_grid(patches,valid):
    """Fixed targets cannot collapse with learned O pooling. Padding excluded by area."""
    b,n,d=patches.shape;assert n==196
    x=patches.float().transpose(1,2).reshape(b,d,14,14)
    m=F.avg_pool2d(valid[:,None].float(),16,16)
    z=F.adaptive_avg_pool2d(x*m,(2,4))/F.adaptive_avg_pool2d(m,(2,4)).clamp_min(1e-6)
    return z.flatten(2).transpose(1,2)
def split_sentences(text,tokenizer,max_tokens=64):
    """No truncation: sentence chunks, then token-bounded word chunks for long sentences."""
    budget=max_tokens-tokenizer.num_special_tokens_to_add(pair=False)
    sentences=[s.strip() for s in re.split(r'(?<=[.!?。！？])\s*',text.strip()) if s.strip()]
    result=[];buf=''
    def length(s):return len(tokenizer.encode(s,add_special_tokens=False))
    for sentence in sentences:
        if length(sentence)>budget:
            if buf:result.append(buf);buf=''
            piece=''
            # Whitespace words; CJK/no-space fallback splits characters without dropping text.
            words=sentence.split() if ' ' in sentence else list(sentence)
            for word in words:
                sep=' ' if ' ' in sentence else ''
                candidate=piece+sep+word if piece else word
                if length(candidate)>budget and piece:result.append(piece);piece=word
                else:piece=candidate
            if piece:result.append(piece)
        elif buf and length(buf+' '+sentence)>budget:result.append(buf);buf=sentence
        else:buf=(buf+' '+sentence).strip()
    if buf:result.append(buf)
    if not result:result=['']
    if any(length(s)>budget for s in result):raise ValueError('Unsplit token exceeds text context')
    return result
class FrozenEncoder(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__();from transformers import AutoModel,AutoTokenizer
        self.model=AutoModel.from_pretrained(MODEL,local_files_only=True).eval().to(device);self.model.requires_grad_(False)
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,use_fast=True)
        self.device=device
    def train(self,mode=True):super().train(False);self.model.eval();return self
    @torch.no_grad()
    def images(self,pixels,valid):
        with torch.autocast('cuda',dtype=torch.float16,enabled=str(self.device).startswith('cuda')):
            out=self.model.vision_model(pixel_values=pixels.to(self.device)).last_hidden_state
        # fp16 round-trip is identical for online and cache paths.
        return spatial_grid(out,valid.to(self.device)).half().float()
    @torch.no_grad()
    def text(self,text):
        chunks=split_sentences(text.lower(),self.tokenizer)
        inputs=self.tokenizer(chunks,padding='max_length',max_length=64,truncation=False,return_tensors='pt').to(self.device)
        out=self.model.get_text_features(**inputs)
        if not isinstance(out,torch.Tensor):out=out.pooler_output
        return out.float().half().float()
