import json,time,torch,numpy as np
from e99w2_common import *
from e99w2_data import load
from e99w2_model import WorldModel,trajectory,JOINTS
from e102a_contract import contract,onset_labels
from e99w2_complete_h100 import UnionImage,EXTRA
from e88.audit import HorizonIndex
from e96a.train import configure
from e86.data import move
from e100c_resource import Guard

def match(pred,true,tol):
 # DP: maximum one-to-one same-direction cardinality, then minimum absolute error.
 n,m=len(pred),len(true);dp=[[None]*(m+1) for _ in range(n+1)];dp[0][0]=(0,0,[])
 def better(old,new):return new if old is None or (new[0],-new[1])>(old[0],-old[1]) else old
 for i in range(n+1):
  for j in range(m+1):
   v=dp[i][j]
   if v is None:continue
   if i<n:dp[i+1][j]=better(dp[i+1][j],v)
   if j<m:dp[i][j+1]=better(dp[i][j+1],v)
   if i<n and j<m and pred[i][1]==true[j][1] and abs(pred[i][0]-true[j][0])<=tol:
    err=pred[i][0]-true[j][0];dp[i+1][j+1]=better(dp[i+1][j+1],(v[0]+1,v[1]+abs(err),v[2]+[err]))
 return dp[n][m][2]
def events(s,initial):
 s=np.concatenate([[initial],s]);idx=np.flatnonzero(s[1:]!=s[:-1])+1;return [(int(j),int(s[j])) for j in idx]
def evaluate():
 configure();torch.backends.mha.set_fastpath_enabled(False);_,_,tr,dev=load();f=json.loads((CONTROL/'fixed_windows.json').read_text());c=contract()
 assert len(f['train64'])==64 and len(f['dev'])==48
 c0=json.loads((dev.base.cache/'contract.json').read_text());c1=json.loads((EXTRA/'contract.json').read_text());assert c0['preprocess']==c1['preprocess'] and c0['encoder_files']==c1['encoder_files'];assert json.loads((T/'results/e93_dev_diag_20261114_01/cache_replay.json').read_text())['max_abs_difference']==0
 dev.base.__class__=UnionImage;dev.index=HorizonIndex(EXTRA,list(dev.index.captures.values()))
 cp=torch.load(RUN/'ckpt_02000.pt',map_location='cpu',weights_only=False);assert cp['step']==STEPS;m=WorldModel(tr.norm);m.load_state_dict(cp['model']);del cp
 guard=Guard('eval',300);actions=[];world=[];latency={};cache={}
 try:
  m=m.to('cuda').eval()
  with torch.inference_mode():
   for split,data,windows in [('dev',dev,f['dev']),('train64',tr,f['train64'])]:
    for entry in windows:
     i=entry['index'];r=data.rows[i];assert r['canonical_id']==entry['cid'] and r['anchor']==entry['anchor'] and r['parent_component']==entry['parent'];guard.check();b=move(data.batch([i],(10,),'short'),'cuda');pred=m.infer(b,'action_only',4,90)[0];assert torch.isfinite(pred).all();pp=pred[0].cpu().numpy();truth=b['action'][0].cpu().numpy();copy=b['q_current'][0].cpu().numpy();labels=b['grip_label'][0].cpu().numpy().astype('i1');current=b['grip_current'][0].cpu().numpy().astype('i1');state=(pp[:,[6,13]]>np.asarray(c['threshold_m'])).astype('i1');z,q,p,g=data.states.get(data,r['canonical_id'])
     key=(split,r['canonical_id'])
     if key not in cache:cache[key]=onset_labels(q,z['continuous_segment'],g,c)[0]
     init=cache[key][r['anchor']];item=dict(split=split,index=i,parent=r['parent_component'],state_accuracy_by_gripper=(state==labels).astype(float).tolist(),copy_accuracy_by_gripper=(current[None]==labels).astype(float).tolist(),state_accuracy=(state==labels).mean(1).tolist(),copy_accuracy=(current[None]==labels).mean(1).tolist(),joints={},events={})
     for h in (8,50):
      err=pp[:h,JOINTS]-truth[:h,JOINTS];ce=copy[JOINTS]-truth[:h,JOINTS];item['joints'][str(h)]=dict(l1=np.abs(err).mean(0).tolist(),mse=(err**2).mean(0).tolist(),copy_l1=np.abs(ce).mean(0).tolist(),copy_mse=(ce**2).mean(0).tolist())
     for tol in (5,10):
      pred_e=[];true_e=[];errs=[]
      for k in range(2):
       pe=events(state[:,k],current[k]);te=events(labels[:,k],init[k]);pred_e+=pe;true_e+=te;errs+=match(pe,te,tol)
      item['events'][str(tol)]=dict(tp=len(errs),pred=len(pred_e),true=len(true_e),errors=errs,minutes=50/30/60,copy_pred=0)
     actions.append(item)
     if split=='dev':
      for h in (25,50,100):
       bh=move(dev.batch([i],(h,),'short'),'cuda');assert bh['world_valid'].all();no=dict(bh,robot=torch.zeros_like(bh['robot']),robot_valid=torch.zeros_like(bh['robot_valid']));oracle,valid=dev.oracle(i)
       # Raw oracle is only a diagnostic; quantize gripper through trajectory via actual observed sequence.
       z,qq,_,_=dev.states.get(dev,r['canonical_id']);a=r['anchor'];seq=qq[a+1:a+101];assert len(seq)==100
       # Reconstruct all101 physical states, preserving oracle joints and fixed two-state grips.
       from e102a_contract import fixed_np
       from e99w2_data import pose_np
       ctx=bh['context'][0].cpu().numpy();seq=np.concatenate([ctx[None,:14],fixed_np(seq,c)]);p0=ctx[84:102];feat=np.concatenate([np.broadcast_to(ctx[:70],(101,70)),seq-ctx[:14],np.broadcast_to(p0,(101,18)),pose_np(seq)-p0,np.broadcast_to(ctx[120:124],(101,4))],1).astype('f4')
       sources=dict(gt_action=bh,no_traj=no,pred_N1=dict(bh,robot=trajectory(bh['context'],pred)),oracle=dict(bh,robot=torch.tensor(feat[None],device='cuda'),robot_valid=valid[None].to('cuda')))
       Y=bh['world'].double()
       for method,bb in sources.items():
        P=m.forward_mean(bb)['mean'].double();assert torch.isfinite(P).all();world.append(dict(index=i,parent=r['parent_component'],h=h,method=method,model=float((P-Y).square().mean()),copy=float(Y.square().mean())))
    print('actions',split,len(windows),flush=True)
   b=move(dev.batch([f['dev'][0]['index']],(25,50,100),'short'),'cuda')
   for mode in ('action_only','joint'):
    for _ in range(10):m.infer(b,mode,4,90)
    times=[]
    for _ in range(100):
     guard.check();torch.cuda.synchronize();start=time.perf_counter();m.infer(b,mode,4,90);torch.cuda.synchronize();times.append(1000*(time.perf_counter()-start))
    latency[mode]=dict(batch=1,n=100,median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),includes='model infer only, cached features; no IO/encoder')
 finally:guard.finish()
 atomic(ART/'evaluation.json',dict(group=GROUP,actions=actions,world=world,latency=latency,checkpoint_sha=sha(RUN/'ckpt_02000.pt'),brt=brt(),event_note='forecast-window exposure denominator; overlapping windows can repeat events; predicted transitions include observed current->first output; ground-truth events use onset labels at anchor, so ongoing pre-anchor motion may cause boundary false transitions'))
 print('evaluation complete',GROUP,len(actions),len(world),flush=True)
if __name__=='__main__':evaluate()
