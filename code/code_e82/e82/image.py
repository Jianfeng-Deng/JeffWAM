"""CPU geometry. RGB float [0,1], explicit original->network affine and valid pixels."""
import numpy as np
import torch

def prepare(rgb,size,box=None):
    import cv2
    h,w=rgb.shape[:2]; x0,y0,x1,y1=(0,0,w,h) if box is None else box
    x0,y0,x1,y1=map(int,(x0,y0,x1,y1))
    if not (0<=x0<x1<=w and 0<=y0<y1<=h): raise ValueError('Invalid original-image crop')
    im=rgb[y0:y1,x0:x1]; nh,nw=size; s=min(nw/im.shape[1],nh/im.shape[0])
    rh,rw=round(im.shape[0]*s),round(im.shape[1]*s); py=(nh-rh)//2; px=(nw-rw)//2
    resized=cv2.resize(im,(rw,rh),interpolation=cv2.INTER_AREA)
    out=cv2.copyMakeBorder(resized,py,nh-rh-py,px,nw-rw-px,cv2.BORDER_REPLICATE)
    mask=np.zeros((nh,nw),np.float32); mask[py:py+rh,px:px+rw]=1
    affine=np.array([[rw/(x1-x0),0,px-x0*rw/(x1-x0)],[0,rh/(y1-y0),py-y0*rh/(y1-y0)]],np.float32)
    return torch.from_numpy(out.copy()).permute(2,0,1).float()/255,torch.from_numpy(mask)[None],affine
