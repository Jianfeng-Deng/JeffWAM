import torch,numpy as np,collections
from e86.data import GRIPS,JOINTS,collate,move
from e86.inference import predict

def signed_current(dataset,row):
    z=np.load(dataset.captures[row['canonical_id']]['numeric_cache']);a=row['anchor'];q=z['q'][a].copy()
    if 'qpos_raw' in z:q[GRIPS]=np.where(z['qpos_raw'][a,GRIPS]<0,z['qpos_raw'][a,GRIPS],q[GRIPS])
    return q
@torch.no_grad()
def action_evaluate(m,ds,device,guard,world_data=None):
    m.eval();records=[]
    for i,row in enumerate(ds.rows):
        guard.check();b=move(collate([ds[i]]),device);y=b['action'][0].cpu().numpy();qlegacy=(b['q_normalized'][0].cpu().numpy()*np.array(ds.stats['q_std'])+np.array(ds.stats['q_mean']));qs=signed_current(ds,row)
        preds={'copy_current_signed':np.broadcast_to(qs,y.shape),'copy_e86_q':np.broadcast_to(qlegacy,y.shape),'action_only':predict(m,b)[0].cpu().numpy()}
        for mode,pred in preds.items():
            err=abs(pred-y);records.append(dict(parent=row['parent_component'],stratum=row.get('stratum','unknown'),mode=mode,curve_grip_mm=err[:,GRIPS].mean(-1)*1000,first8_grip_mm=float(err[:8,GRIPS].mean()*1000),H50_grip_mm=float(err[:,GRIPS].mean()*1000),first8_joint_rad=float(err[:8,JOINTS].mean()),H50_joint_rad=float(err[:,JOINTS].mean())))
    def aggregate(rs):
        out={}
        for mode in sorted({r['mode'] for r in rs}):
            selected=[r for r in rs if r['mode']==mode];ps=sorted({r['parent'] for r in selected});out[mode]={key:np.mean([np.mean([r[key] for r in selected if r['parent']==p],axis=0) for p in ps],axis=0).tolist() for key in ('curve_grip_mm','first8_grip_mm','H50_grip_mm','first8_joint_rad','H50_joint_rad')}
        return out
    return dict(windows=len(ds),parents=len({r['parent_component'] for r in ds.rows}),parent_equal=True,summary=aggregate(records),strata={st:aggregate([r for r in records if r['stratum']==st]) for st in sorted({r['stratum'] for r in records})},truth='same original signed next-frame feedback, no model language truth',checkpoint_selection=False)
@torch.no_grad()
def world_evaluate(m,data,ids,device,guard,language='short',steps=4):
    m.eval();records=[]
    for i in ids:
        row=data.rows[i]
        for h in (10,25,50,100,200,300):
            b=data.batch([i],(h,),language);valid=b['world_valid']
            if not valid.any():continue
            guard.check();b=move(b,device);pred=m.rollout(b,steps,seed=8801);cur=b['world_current'];truth=b['world_raw'];scale=b['world_scale']
            # causal linear extrapolation from current and last valid real historical frame, using source time.
            hv=b['history_valid'][0];js=hv.nonzero().flatten();prev=b['obs'][0,int(js[-1])] if len(js) else b['obs'][0,4]
            z,_=data.index.get(row['canonical_id']);ct=z['camera_exposure_timestamp'];future_ids,_=data.index.resolve(row,h);ratios=[]
            for cam in range(3):
                nowrow=row['obs_rows'][4][cam];pastrow=row['obs_rows'][int(js[-1])][cam] if len(js) else nowrow
                dt=ct[nowrow,cam]-ct[pastrow,cam];ratios.append((ct[future_ids[cam],cam]-ct[nowrow,cam])/dt if dt>1e-7 else 0.)
            if h>50:prev=prev.mean(1,keepdim=True)
            prev=prev.reshape(1,-1,768);n=8 if h<=50 else 1;ratio=torch.tensor(ratios,device=device,dtype=cur.dtype)[:,None].expand(3,n).reshape(1,-1,1)
            linear=cur+(cur-prev)*ratio
            mse=lambda z:float((((z-truth)/scale).square().mean(-1)*b['world_valid']).sum()/b['world_valid'].sum())
            modelerr=mse(pred);copy=mse(cur);lin=mse(linear);records.append(dict(h=h,parent=row['parent_component'],stratum=row['stratum'],model=modelerr,copy=copy,linear=lin))
    def agg(rs):
        if not rs:return dict(windows=0,parents=0,model=None,copy=None,linear=None,skill_copy=None,skill_linear=None)
        ps=sorted({r['parent'] for r in rs});d={k:float(np.mean([np.mean([r[k] for r in rs if r['parent']==p]) for p in ps])) for k in ('model','copy','linear')};return dict(windows=len(rs),parents=len(ps),**d,skill_copy=1-d['model']/max(d['copy'],1e-12),skill_linear=1-d['model']/max(d['linear'],1e-12))
    return {str(h):dict(overall=agg([r for r in records if r['h']==h]),strata={st:agg([r for r in records if r['h']==h and r['stratum']==st]) for st in sorted({r['stratum'] for r in records})}) for h in (10,25,50,100,200,300)}
