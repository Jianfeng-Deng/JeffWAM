"""CPU source+cache availability, causal labels, gripper scale and tiny-head audit."""
import json,collections,time,numpy as np,torch
from pathlib import Path
from e86.data import Package,CachedDataset,GRIPS,collate
from e86.common import sha,atomic
from .common import *
HORIZONS=(10,25,50,100,200,300)
def read_inputs():
    from e86.data import verify_receipt
    p=Package();r=verify_receipt(INPUT/'READY_V3_cache_windows.json')
    if p.refs!=r['refs']:raise ValueError('Package drift')
    ws={s:json.loads(Path(r['windows'][s]).read_text()) for s in ('P1','P3')}
    stats=json.loads((INPUT/'stats.json').read_text())
    return p,r,ws,stats
class HorizonIndex:
    def __init__(self,cache,captures):
        self.cache=Path(cache);self.captures={c['canonical_id']:c for c in captures};self.memo={}
        self.roots=[self.cache]
        if (self.cache/'overlay.json').exists():self.roots.append(Path(json.loads((self.cache/'overlay.json').read_text())['base_cache']))
    def get(self,cid):
        if cid not in self.memo:
            z=dict(np.load(self.captures[cid]['numeric_cache'],allow_pickle=False));sets=[]
            for cam in ('cam_high','cam_left_wrist','cam_right_wrist'):
                parts=[np.load(root/cid/(cam+'_rows.npy')) for root in self.roots if (root/cid/(cam+'_rows.npy')).exists()]
                vals=np.concatenate(parts) if parts else np.array([],int)
                if len(np.unique(vals))!=len(vals):raise ValueError('Duplicate overlay rows')
                sets.append(set(map(int,vals)))
            self.memo[cid]=(z,sets)
        return self.memo[cid]
    def resolve(self,row,h,require_cached=True):
        z,sets=self.get(row['canonical_id']);a=row['anchor'];f=a+int(h);n=len(z['timestamp'])
        if f>=n:return None,'end'
        lo,hi=row.get('subinterval_raw_rows_exclusive') or (0,n)
        if not lo<=a<=f<hi:return None,'subinterval'
        if z['continuous_segment'][a]!=z['continuous_segment'][f]:return None,'seam'
        mask=z['p1_source_mask'] if row.get('_stage','P3')=='P1' else z['p3_source_mask']
        if not mask[a:f+1].all():return None,'source_mask'
        if 'original_observation_index' in z and not np.array_equal(z['original_observation_index'][a:f+1],np.arange(a,f+1)):return None,'source_index'
        target=z['timestamp'][f];rr=np.arange(max(lo,f-2),min(hi,f+3));out=[]
        for k in range(3):
            ct=z['camera_exposure_timestamp'];j=int(rr[np.argmin(abs(ct[rr,k]-target))])
            if z['continuous_segment'][j]!=z['continuous_segment'][a] or abs(ct[j,k]-target)>1/60+1e-6:return None,'camera_time_or_seam'
            if require_cached and j not in sets[k]:return None,'cache'
            out.append(j)
        return out,None

def labels_index(p):
    path=R/'lang_codex/A/lang_A.jsonl';index=collections.defaultdict(list)
    for r in rows(path):
        if r['split']!='train' or p.split.get(r['parent_id'])!='train':raise ValueError('Eval model narration')
        index[r['canonical_id']].extend(r.get('process_narration',[]))
    return index,dict(path=str(path),sha256=sha(path))
def causal_segments(segments,anchor):
    # Unknown/missing cutoff is ineligible; segment end is not an implicit cutoff.
    return [(i,x) for i,x in enumerate(segments) if isinstance(x.get('causal_evidence_cutoff_source_frame'),(int,float)) and x['causal_evidence_cutoff_source_frame']<=anchor and x['source_frame_start']<=anchor]
def stage_target(segments,anchor):
    xs=[(i,x) for i,x in causal_segments(segments,anchor) if x['source_frame_start']<=anchor<=x['source_frame_end']]
    if not xs:return -1,0.,False
    i,x=xs[-1];return i,min(1.,max(0.,(anchor-x['source_frame_start'])/max(1,x['source_frame_end']-x['source_frame_start']))),True

def main():
    torch.set_num_threads(2);torch.manual_seed(88);dest=ART/'cpu_audit_01';dest.mkdir(exist_ok=False);p,r,ws,stats=read_inputs();ix=HorizonIndex(r['cache'],p.captures);li,lr=labels_index(p);summary={};coverage={};tables={}
    for stage,rs in ws.items():
        table=np.full((len(rs),6,3),-1,np.int32);counts=collections.Counter();parents=collections.defaultdict(set);tot=collections.Counter();allparents=collections.defaultdict(set);cov=collections.defaultdict(collections.Counter)
        for i,row in enumerate(rs):
            st=row['stratum'];tot[st]+=1;allparents[st].add(row['parent_component']);anchor=row['anchor'];segments=li[row['canonical_id']]
            cov[st]['windows']+=1;cov[st]['any_narration']+=bool(segments);cov[st]['causal_text']+=bool(causal_segments(segments,anchor));cov[st]['stage_progress']+=stage_target(segments,anchor)[2]
            for j,h in enumerate(HORIZONS):
                v,why=ix.resolve(dict(row,_stage=stage),h)
                if v is not None:
                    table[i,j]=v;counts[(st,h,'valid')]+=1;parents[(st,h)].add(row['parent_component'])
                else:counts[(st,h,why)]+=1
            if i%10000==0:atomic(dest/'status.json',dict(stage=stage,windows_done=i,total=len(rs),status='CPU_availability'))
        np.savez_compressed(dest/(stage+'_horizon_rows.npz'),rows=table);tables[stage]=table
        summary[stage]={str(h):dict(windows=int((table[:,j,0]>=0).sum()),parents=len({rs[i]['parent_component'] for i in np.flatnonzero(table[:,j,0]>=0)}),strata={st:dict(windows=counts[(st,h,'valid')],parent_coverage=len(parents[(st,h)]),total_windows=tot[st],total_parents=len(allparents[st]),reasons={why:v for (ss,hh,why),v in counts.items() if ss==st and hh==h and why!='valid'}) for st in sorted(tot)}) for j,h in enumerate(HORIZONS)}
        coverage[stage]={st:dict(v) for st,v in cov.items()};atomic(dest/'availability.json',summary);atomic(dest/'stage_coverage.json',dict(source=lr,coverage=coverage,strict_current_segment=True,note='current stage is masked unless narrative evidence cutoff <= current anchor; completed past narration may condition text, never inferred current stage'))
    eligible=[h for h in HORIZONS if summary['P3'][str(h)]['windows']/34777>=.6 and summary['P3'][str(h)]['parents']/145>=.9];longest=max(eligible) if eligible else None
    atomic(dest/'recommendation.json',dict(longest_horizon=longest,rule='P3 windows >=60%, parents >=90%',cache_read='per camera closest exposure to timestamp[anchor+h], within +-2 raw rows and <=1/60s; exact cached original row required',source_ready_sha256=sha(INPUT/'READY_V3_cache_windows.json')))
    report('### B/E CPU 全池统计已落盘\n'+json.dumps(dict(recommendation=longest,availability=summary,stage_coverage=coverage),ensure_ascii=False))
    # Train-only residual RMS per target token, parent equal, up to 2 windows/parent/h.
    ds=CachedDataset([],p.captures,r['cache'],stats);norm={};samples={}
    rs=ws['P1'];table=tables['P1']
    for j,h in enumerate(HORIZONS):
        by=collections.defaultdict(list)
        for i in np.flatnonzero(table[:,j,0]>=0):by[rs[i]['parent_component']].append(int(i))
        ms=[];records=[]
        for parent,ids in sorted(by.items()):
            vals=[]
            for i in [ids[k] for k in np.linspace(0,len(ids)-1,min(2,len(ids))).astype(int)]:
                row=rs[i];cid=row['canonical_id'];cur=np.stack([ds.features(cid,k,[row['obs_rows'][-1][k]])[0] for k in range(3)]);future=np.stack([ds.features(cid,k,[table[i,j,k]])[0] for k in range(3)])
                delta=future-cur
                if h>50:delta=delta.mean(1,keepdims=True)
                vals.append((delta**2).mean(-1));records.append([cid,row['anchor'],parent])
            ms.append(np.mean(vals,0))
        norm[str(h)]=np.maximum(np.sqrt(np.mean(ms,0)),.05).tolist();samples[str(h)]=records
    atomic(dest/'residual_stats.json',dict(scale=norm,floor=.05,statistic='sqrt(parent_equal_mean(window_mean(channel_mean(delta^2)))) per camera/token',fitted_split='train',stage='P1',sampling='deterministic up to2 windows per train parent per horizon',samples=samples,refs=p.refs))
    # Exact additivity on identical fixed encoded endpoints (algebraic identity).
    diag=[]
    for i,row in enumerate(ws['P1']):
        if len(diag)>=128:break
        if (tables['P1'][i,1:,0]>=0).all():
            cid=row['canonical_id'];cur=np.stack([ds.features(cid,k,[row['obs_rows'][-1][k]])[0] for k in range(3)]);m=np.stack([ds.features(cid,k,[tables['P1'][i,1,k]])[0] for k in range(3)]);end=np.stack([ds.features(cid,k,[tables['P1'][i,2,k]])[0] for k in range(3)])
            a=(end-cur).ravel();b=((m-cur)+(end-m)).ravel();diag.append((float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12)),float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))))
    atomic(dest/'additivity.json',dict(windows=len(diag),cosine=float(np.mean([x[0] for x in diag])),relative_error=float(np.mean([x[1] for x in diag])),interpretation='fixed encoder differences telescope algebraically; necessary sanity check, not evidence predicted rollouts are accurate',endpoint='same exact camera rows for shared midpoint'))
    gripper_audit(dest,p,r,ws,stats)
    atomic(dest/'status.json',dict(status='completed',updated_brt=brt(),cuda_initialized=torch.cuda.is_initialized()))

def gripper_audit(dest,p,ready,ws,stats):
    # Deterministic spread across train parents/strata, no train window-ID features.
    selected=[];by=collections.defaultdict(list)
    for row in ws['P3']:by[row['parent_component']].append(row)
    for parent,vs in sorted(by.items()):
        selected.append(vs[len(vs)//2])
        if len(selected)==64:break
    devcaps=json.loads((C/'artifacts/dev_v1/captures.json').read_text());devws=json.loads((C/'artifacts/dev_v1/windows_v2.json').read_text());datasets={'train64':CachedDataset(selected,p.captures,ready['cache'],stats),'dev48':CachedDataset(devws,devcaps,C/'artifacts/dev_features_v1',stats)}
    result={};batches={}
    for name,ds in datasets.items():
        b=collate([ds[i] for i in range(len(ds))]);batches[name]=b
        q=b['q_normalized']*torch.tensor(stats['q_std'])+torch.tensor(stats['q_mean']);err=(b['action'][:,:,GRIPS]-q[:,None,GRIPS]).abs()*1000
        # Verify signed-current difference separately, never change feedback labels.
        signed=[]
        for row in ds.rows:
            z=np.load(ds.captures[row['canonical_id']]['numeric_cache']);a=row['anchor'];raw=z.get('qpos_raw',z['q'])[a,GRIPS];current=z['q'][a,GRIPS].copy();current=np.where(raw<0,raw,current);signed.append(current)
        signed=torch.tensor(np.array(signed));signederr=(b['action'][:,:,GRIPS]-signed[:,None]).abs()*1000
        rec=lambda x:dict(curve_mm=x.mean((0,2)).tolist(),first8_mm=float(x[:,:8].mean()),H50_mm=float(x.mean()))
        result[name]=dict(copy_e86=rec(err),copy_signed_current=rec(signederr),signed_vs_q_max_mm=float((signed-q[:,GRIPS]).abs().max()*1000),anchor_grips_zero=bool((b['q_anchor'][:,GRIPS]==0).all()),std_range_m=[float(b['action_std'][:,:,GRIPS].min()),float(b['action_std'][:,:,GRIPS].max())],target_grips_m=[float(b['action'][:,:,GRIPS].min()),float(b['action'][:,:,GRIPS].max())],roundtrip_max_m=float(((b['action']-b['q_anchor'][:,None]-b['action_mean'])/b['action_std']*b['action_std']+b['action_mean']+b['q_anchor'][:,None]-b['action']).abs().max()))
    b=batches['train64'];x=torch.cat((b['obs'][:,4].mean((1,2)),b['q_normalized']),1);x=(x-x.mean(0))/x.std(0).clamp_min(.1);x=x.detach();target=b['action'][:,:,GRIPS];mean=b['action_mean'][:,:,GRIPS];scale=b['action_std'][:,:,GRIPS];y=((target-mean)/scale).reshape(64,-1).detach()
    head=torch.nn.Sequential(torch.nn.Linear(x.shape[1],256),torch.nn.GELU(),torch.nn.Linear(256,100));opt=torch.optim.Adam(head.parameters(),lr=.002);history=[];start=time.monotonic()
    for step in range(3000):
        pred=head(x);v=(pred-y).square().mean();opt.zero_grad();v.backward();opt.step()
        if step%100==0:history.append([step,float(v.detach())])
        if v<1e-7:break
    with torch.no_grad():pred=head(x).reshape(64,50,2)*scale+mean;err=(pred-target).abs()*1000
    result['head_only_overfit']=dict(windows=64,steps=step+1,elapsed_s=time.monotonic()-start,loss_history=history,first8_mm=float(err[:,:8].mean()),H50_mm=float(err.mean()),curve_mm=err.mean((0,2)).tolist(),below_copy=bool(err.mean()<torch.tensor(result['train64']['copy_e86']['H50_mm'])*.25),head='isolated deterministic absolute gripper MLP; frozen raw current SigLIP2 mean + current q; no IDs; not full FM model',no_dev_fitting=True)
    atomic(dest/'gripper_cpu.json',result);report('### A CPU 尺度审计与64窗夹爪头过拟合\n'+json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
