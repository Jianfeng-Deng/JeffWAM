"""Recorded next-frame feedback errors; no language scores or checkpoint selection."""
import torch,numpy as np
from collections import defaultdict
from .data import JOINTS,GRIPS,collate,move
from .inference import predict
@torch.no_grad()
def evaluate(model,dataset,device,guard=None):
    model.eval();values=defaultdict(lambda:defaultdict(list));strata=defaultdict(lambda:defaultdict(lambda:defaultdict(list)));perwindow=[]
    # Per-window inference keeps fixed noise independent of batch grouping.
    for i,row in enumerate(dataset.rows):
        if guard:guard.check()
        b=move(collate([dataset[i]]),device);target=b['action'][0].float().cpu().numpy();q=b['q_anchor'][0].cpu().numpy().copy();q[GRIPS]=b['q_normalized'][0,GRIPS].cpu().numpy()*np.array(dataset.stats['q_std'])[GRIPS]+np.array(dataset.stats['q_mean'])[GRIPS]
        baseline=np.broadcast_to(q,(50,14)).copy();predictions={'copy_current_state':baseline}
        for mode in ('action_only','joint'):predictions[mode]=predict(model,b,mode)[0].float().cpu().numpy()
        nolang=dict(b,text_valid=torch.zeros_like(b['text_valid']));predictions['no_language_inference']=predict(model,nolang,'action_only')[0].float().cpu().numpy()
        for name,p in predictions.items():
            for horizon in (8,50):
                err=p[:horizon]-target[:horizon];j=err[:,JOINTS];gr=err[:,GRIPS]
                metrics=dict(joint_L1_rad=float(np.abs(j).mean()),joint_angle_deg=float(np.abs(j).mean()*180/np.pi),gripper_L1_mm=float(np.abs(gr).mean()*1000),left_joint_L1_rad=float(np.abs(err[:,:6]).mean()),right_joint_L1_rad=float(np.abs(err[:,7:13]).mean()),left_gripper_L1_mm=float(np.abs(err[:,6]).mean()*1000),right_gripper_L1_mm=float(np.abs(err[:,13]).mean()*1000))
                for k,v in metrics.items():
                    values[name+'_'+str(horizon)][row['parent_component']].append((k,v))
                    strata[row.get('stratum','unknown')][name+'_'+str(horizon)][row['parent_component']].append((k,v))
        perwindow.append(dict(canonical_id=row['canonical_id'],anchor=row['anchor'],parent=row['parent_component'],truth='recorded_feedback'))
    summary={}
    for name,parents in values.items():
        buckets=defaultdict(list)
        for p,vals in parents.items():
            by=defaultdict(list)
            for k,v in vals:by[k].append(v)
            for k,v in by.items():buckets[k].append(float(np.mean(v)))
        summary[name]={k:float(np.mean(v)) for k,v in buckets.items()}
    stratified={}
    for st,groups in strata.items():
        out={}
        for name,parents in groups.items():
            metrics_by_parent=defaultdict(list)
            for p,vals in parents.items():
                by=defaultdict(list)
                for k,v in vals:by[k].append(v)
                for k,vs in by.items():metrics_by_parent[k].append(float(np.mean(vs)))
            out[name]={k:float(np.mean(vs)) for k,vs in metrics_by_parent.items()}
        selected=[r for r in dataset.rows if r.get('stratum','unknown')==st]
        stratified[st]=dict(windows=len(selected),parents=len({r['parent_component'] for r in selected}),results=out)
    return dict(strata=stratified,windows=len(dataset),parents=len(set(r['parent_component'] for r in dataset.rows)),parent_equal=True,results=summary,truth='unchanged next-frame recorded follower feedback; not sent command',scope='smoke only; fixed stage-end; no checkpoint choice; no model language annotations on dev',windows_manifest=perwindow)
