from e107m_test import *
configure();torch.backends.mha.set_fastpath_enabled(False);_,_,tr,_=load();cp=torch.load(CK,map_location='cpu',weights_only=False);a=WorldModel(tr.norm);a.load_state_dict(cp['model']);b=StochasticWorld(tr.norm);b.load_state_dict(cp['model'],strict=False);a.eval();b.eval();d=tr.batch([0,1],(10,25,50,100),'short')
for n,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[n]),n
with torch.no_grad():
 for tag in ('normal','freeze','hook'):
  if tag=='freeze':a.requires_grad_(False)
  if tag=='hook':a.mean_head.register_forward_hook(lambda mod,args,result:result)
  x=a.forward_mean(d)['mean'];xx=a.forward_mean(d)['mean'];y=b.forward_mean(d)['mean'];yy=b.forward_mean(d)['mean'];print(tag,'repeat a/b',torch.equal(x,xx),torch.equal(y,yy),'cross',torch.equal(x,y),'err',float((x-y).abs().max()),flush=True)
