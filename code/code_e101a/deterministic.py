import os,sys
sys.dont_write_bytecode=True
sys.path.insert(0,'/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e101a')
from diag import *
status('loading');_,_,tr,dev=load();cp=torch.load(T/'results/e100c_c2_s90_20261115_03/ckpt_05000.pt',map_location='cpu',weights_only=False);m=WorldModel(tr.norm);m.load_state_dict(cp['model']);del cp;m.eval();torch.backends.mha.set_fastpath_enabled(False)
meta=json.loads((OUT/'fixed_windows.json').read_text());res={};pred={}
with torch.inference_mode():
 for name,d in [('dev',dev),('train64',tr)]:
  aa=[]
  for k,r in enumerate(meta[name]):
   b=d.batch([r['index']],(10,),'short');z=torch.zeros(1,50,14);t=torch.zeros(1);v=m(b,z,None,t,t)['action'];a=b['action_mean']+v*b['action_std']+b['q_anchor'][:,None];aa.append(a[0].numpy())
   if k%16==0:status('infer_'+name,done=k)
  dat=np.load(OUT/(name+'.npz'));pred[name]=np.array(aa);res[name]=metrics(pred[name],dat['y'],dat['q'],dat['parents'])
np.savez_compressed(OUT/'t0_predictions.npz',**pred);save('t0_metrics.json',res);status('complete',cuda_initialized=torch.cuda.is_initialized())
