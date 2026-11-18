import copy,json,torch,numpy as np
from e99w2_common import *
from e99w2_data import load
from e99w2_model import WorldModel
from e100c_chain import reentered,corrected_chain,open_second
from e96a.train import configure
from e86.data import collate

def main():
 import e99w2_train,e99w2_objective,e96a.train
 assert Path(e99w2_train.__file__).resolve().parent==ROOT
 assert Path(e99w2_objective.__file__).resolve().parent==ROOT
 assert Path(e96a.train.__file__).resolve().parent.parent==ROOT
 configure();torch.backends.mha.set_fastpath_enabled(False)
 _,_,tr,_=load();m=WorldModel(tr.norm);m.load_state_dict(torch.load(INIT,map_location='cpu',weights_only=False)['model'],strict=True);m.eval();results=[]
 for k in (25,50):
  for i in range(500):
   teacher=tr.shifted(i,k)
   if teacher is not None and teacher['world_valid'].all() and tr.index.resolve(tr.rows[i],2*k)[0] is not None:break
  else:raise AssertionError('No legal CPU test window')
  first=tr.batch([i],(k,));direct=tr.batch([i],(2*k,));teacher=collate([teacher]);teacher['world_positions']=teacher['world_positions'][0]
  with torch.no_grad():
   a,_=reentered(m,first,teacher,k,True);y=m.mean_rollout(a)
   # Every real shifted frame, measured state, metadata, language and target is adversarially changed.
   bad={key:(v.clone() if torch.is_tensor(v) else v) for key,v in teacher.items()}
   for key in ('obs','obs_meta','context','robot','q_current','q_normalized','q_anchor','action','world_current','world_raw','world','align_text','text'):
    if key in bad:bad[key]=torch.randn_like(bad[key])*100
   bad['history_valid']=~bad['history_valid']
   b,_=reentered(m,first,bad,k,True);yp=m.mean_rollout(b);assert torch.equal(y,yp),'Future leakage'
   assert not b['history_valid'].any() and not b['obs'].any()
   assert torch.equal(b['action'],torch.cat([first['action'][:,k:],first['action'][:,-1:].expand(-1,k,-1)],1))
   ta,_=reentered(m,first,teacher,k,False);tb,_=reentered(m,first,bad,k,False);delta=float((m.mean_rollout(ta)-m.mean_rollout(tb)).abs().max());assert delta>0
   # Direct target cannot enter second-hop forward, but must still affect its supervised loss.
   changed=dict(direct,world_raw=direct['world_raw']+100)
   l,_=corrected_chain(m,first,teacher,direct,k,True);l2,_=corrected_chain(m,first,teacher,changed,k,True);assert l!=l2
  m.zero_grad(set_to_none=True);v,metrics=corrected_chain(m,first,teacher,direct,k,True);v.backward();assert m.reentry[0].weight.grad is not None if isinstance(m.reentry,torch.nn.Sequential) else any(p.grad is not None for p in m.reentry.parameters())
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
  results.append(dict(k=k,index=i,predicted_max_abs=float((y-yp).abs().max()),teacher_delta=delta,endpoint_loss=float(v.detach()),gap=metrics['chain_gap'],masked_history=True))
 assert not torch.cuda.is_initialized()
 atomic(CONTROL/'CPU_TESTS.json',dict(status='passed',results=results,CPU_only=True,brt=brt()));report('## CPU开环隔离测试通过\n'+json.dumps(results,ensure_ascii=False));print(json.dumps(results))
if __name__=='__main__':main()
