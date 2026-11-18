import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,collections,hashlib,json
from common import *
from contract import contracts,sha
def rank(s):return hashlib.sha256(('20261112:'+str(s)).encode()).digest()
def prepare(mode='full'):
    split,caps,routes=contracts();out=P/('selection_'+mode+'_v4.json')
    if out.exists():return json.loads(out.read_text())
    from fk_consumption import truth_contract
    truth_contract()
    eligible=json.loads((R/'fk/ELIGIBILITY.json').read_text());bad=json.loads((R/'gt/statistics.json').read_text())['decode_failure_sources']
    excluded=set(eligible['excluded_captures'])|{r['canonical_id'] for r in bad}
    old_captures=split['captures'];split=dict(split,captures=[c for c in old_captures if c['canonical_id'] not in excluded])
    cache={};parents=collections.defaultdict(list);bycap={c['canonical_id']:c for c in split['captures']}
    for c in split['captures']:
        with np.load(c['window_cache']) as w:
            anchors=w['p1_anchor'].astype(int);cache[c['canonical_id']]=anchors
        for wi,a in enumerate(anchors):parents[c['parent_component']].append((c['canonical_id'],wi,int(a)))
    selected=[]
    for which in ['probe-train','probe-val']:
        ps=sorted(p for p in parents if split['parents'][p]==which)
        for pi,p in enumerate(ps):
            choices=parents[p];ids=set(c[0] for c in choices)
            rep=max(ids,key=lambda cid:(len(cache[cid]),cid));aa=cache[rep]
            mandatory=[(rep,int(i),int(aa[i])) for i in np.unique(np.rint(np.linspace(0,len(aa)-1,min(8,len(aa)))).astype(int))]
            quota=len(mandatory) if mode=='cpu' else 12000//len(ps)+int(pi<12000%len(ps))
            pool=sorted(set(choices)-set(mandatory),key=rank)
            chosen=mandatory+pool[:max(0,quota-len(mandatory))]
            for cid,wi,a in chosen:selected.append({'canonical_id':cid,'parent_component':p,'probe_split':which,'cohort':bycap[cid]['cohort'],'anchor_source_row':a,'window_index':wi,'mandatory_gt_coverage':(cid,wi,a) in mandatory})
    selected.sort(key=lambda r:(r['canonical_id'],r['anchor_source_row']))
    grouped=collections.defaultdict(list)
    for row in selected:grouped[row['canonical_id']].append(row)
    for cid,rows in grouped.items():
        with np.load(bycap[cid]['numeric_cache']) as n,np.load(bycap[cid]['window_cache']) as w:
            ts=n['timestamp'];seg=n['continuous_segment'];mask=n['p1_source_mask'];exp=n['camera_exposure_timestamp'];recv=n['camera_received_timestamp']
            for row in rows:
                a=row['anchor_source_row'];wi=row['window_index'];decision=float(w['p1_decision_time'][wi]);camrows=np.full((5,3),-1,dtype=int);ages=np.zeros((5,3),dtype=float)
                for hi,offset in enumerate([.8,.6,.4,.2,0]):
                    clock=decision-offset
                    for ci in range(3):
                        if offset==0: cr=int(w['p1_O_video_rows'][wi,-1,ci])
                        else:
                            valid=np.flatnonzero((exp[:,ci]<=clock)&(recv[:,ci]<=clock)&(seg==seg[a])&mask)
                            cr=int(valid[-1]) if len(valid) else -1
                        if cr>=0:
                            assert mask[cr] and seg[cr]==seg[a]
                            assert exp[cr,ci]<=clock+1e-7 and recv[cr,ci]<=clock+1e-7
                            age=clock-float(exp[cr,ci])
                            if age<=.1+1e-7:camrows[hi,ci]=cr;ages[hi,ci]=age
                assert np.all(camrows[-1]>=0)
                row.update(camera_rows=camrows.tolist(),camera_age_s=ages.tolist(),decision_time=decision,state=n['q'][a].astype(float).tolist(),continuous_segment=int(seg[a]))
                endpoint_rows=np.full((2,3),-1,dtype=int);endpoint_ages=np.zeros((2,3),dtype=float)
                for ei,horizon in enumerate([10,50]):
                    end=a+horizon
                    for ci in range(3):
                        cr=int(w['p1_W_video_rows'][wi,0 if horizon==10 else 2,ci])
                        assert mask[cr] and seg[cr]==seg[a]
                        # Exp2 describes an already observed interval: observation clock waits for
                        # this image receipt. This endpoint is never an input to exp3.
                        clock=max(float(n['state_available_time'][end]),float(exp[cr,ci]),float(recv[cr,ci]))
                        if clock-exp[cr,ci]<=.1+1e-7:
                            endpoint_rows[ei,ci]=cr;endpoint_ages[ei,ci]=clock-exp[cr,ci]
                row.update(endpoint_camera_rows=endpoint_rows.tolist(),endpoint_camera_age_s=endpoint_ages.tolist(),endpoint_semantics='experiment2 only: exact F W10/W50 exposure rows; after-window observation clock waits for image receipt; paired with anchor O; never supplied to experiment3')
    counts=collections.Counter(r['probe_split'] for r in selected)
    obj={'schema':'e84p-selection-v4','mode':mode,'created_brt':brt(),'split_sha256':sha(R/'split/probe_split.json'),'sampling_seed':20261112,'selection':'balanced parents; 8 evenly-spaced longest-capture anchors mandatory; fill by fixed hash, no labels/outcomes; F eligibility and failed-video format gate before model features','counts':dict(counts),'parents':dict(collections.Counter(split['parents'][p] for p in parents)),'excluded_captures':sorted(excluded),'exclusion_reason':'one diagnostic parent prohibited by F; any known undecodable camera capture entirely quarantined, not zero/black placeholder','samples':selected,'source_routes':{cid:routes[cid] for cid in grouped},'processor_contract':'encoder.PROCESSOR; preserve aspect/full FOV, bilinear, explicit affine','estimated_fp16_current_patch_bytes':len(selected)*3*196*768*2,'estimated_fp16_global_bytes':len(selected)*7*3*768*2,'features_read':False}
    atomic(out,obj);update(status='waiting_compute',numeric_metadata_captures_read=len(cache),sample_selection_counts=dict(counts))
    append_report('## 样本预选 '+mode+'\n\n'+brt()+' '+json.dumps({k:obj[k] for k in ['counts','parents','estimated_fp16_current_patch_bytes','estimated_fp16_global_bytes']},ensure_ascii=False)+'。仅允许train数值缓存，当前/历史按曝光与receive双条件因果筛选；不足0.8s历史mask，不跨段。未读图像、不用标签筛选。')
    print(mode,dict(counts),'parents',len(parents),flush=True);return obj
if __name__=='__main__':
    prepare('full');prepare('cpu')
