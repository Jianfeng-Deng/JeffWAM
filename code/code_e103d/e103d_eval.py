import os,json,time,traceback,collections
import numpy as np,torch
from e103d_runtime import *
from e103d_model import WorldModel
from e99w2_data import load
from e99w2_model import trajectory
from e99w2_complete_h100 import UnionImage,EXTRA
from e88.audit import HorizonIndex
from e90.data import Data
from e86.data import move,JOINTS,GRIPS
from e96a.train import configure
H=(10,25,50,100)
def action_record(pred,target,parent,index,method):
 out=[]
 for h in (8,50):
  e=(pred[:h]-target[:h]).double();assert torch.isfinite(e).all();out.append(dict(parent=parent,index=index,method=method,h=h,joint_L1_rad=float(e[:,JOINTS].abs().mean()),joint_RMSE_rad=float(e[:,JOINTS].square().mean().sqrt()),gripper_L1_mm=float(e[:,GRIPS].abs().mean()*1000),gripper_RMSE_mm=float(e[:,GRIPS].square().mean().sqrt()*1000)))
 return out

def evaluate(group):
 configure();torch.backends.mha.set_fastpath_enabled(False);_,_,tr,dev=load();c0=json.loads((dev.base.cache/'contract.json').read_text());c1=json.loads((EXTRA/'contract.json').read_text());assert c0['preprocess']==c1['preprocess'] and c0['encoder_files']==c1['encoder_files'];assert json.loads((T/'results/e93_dev_diag_20261114_01/cache_replay.json').read_text())['max_abs_difference']==0
 dev.base.__class__=UnionImage;dev.index=HorizonIndex(EXTRA,list(dev.index.captures.values()));run=T/f'results/e103d_{group.lower()}_s90_20261115_02';assert json.loads((run/'train_summary.json').read_text())['steps']==5000
 ck=run/'ckpt_05000.pt';cp=torch.load(ck,map_location='cpu',weights_only=False);m=WorldModel(tr.norm);m.load_state_dict(cp['model'],strict=True);del cp;guard=Guard('eval_'+group+'_02',900,6000);records=[];actions=[];expanded=[]
 try:
  m=m.to('cuda').eval()
  with torch.inference_mode():
   for i,r in enumerate(dev.rows):
    guard.check();b=move(dev.batch([i],H,'short'),'cuda');aa=[m.infer(b,'action_only',4,90+n)[0] for n in range(8)]
    for method,a in [('N1',aa[0]),('N8_mean',torch.stack(aa).mean(0))]:actions.extend(action_record(a[0],b['action'][0],r['parent_component'],i,method))
    for h in H:
     bh=move(dev.batch([i],(h,),'short'),'cuda');assert bh['world_valid'].all();Y=bh['world'];no=dict(bh,robot=torch.zeros_like(bh['robot']),robot_valid=torch.zeros_like(bh['robot_valid']));ps={'gt_action':m.forward_mean(bh)['mean'],'no_traj':m.forward_mean(no)['mean']};ap=[m.forward_mean(dict(bh,robot=trajectory(bh['context'],a)))['mean'] for a in aa];ps.update(pred_N1=ap[0],pred_N8=torch.stack(ap).mean(0))
     for method,p in ps.items():
      assert torch.isfinite(p).all();records.append(dict(index=i,parent=r['parent_component'],h=h,method=method,model=float((p.double()-Y.double()).square().mean()),copy=float(Y.double().square().mean()),cosine=float(torch.nn.functional.cosine_similarity(p.flatten(),Y.flatten(),dim=0))))
    if i%12==0:print(group,'fixed',i,flush=True)
   er=T/'results/e93_dev_diag_20261114_01';caps=json.loads((er/'dev/captures_full.json').read_text());rows=json.loads((er/'dev/windows_full.json').read_text());assert all(r['canonical_id']!='piper_4dc107335389f5711a0f65e0' for r in rows);assert not {r['parent_component'] for r in rows}&{r['parent_component'] for r in tr.rows};ed=Data(rows,caps,er/'features',tr.base.stats,{'scale':tr.norm},{'__human_default__':dev.base.text_cache['__human_default__']})
   for i,r in enumerate(ed.rows):
    guard.check()
    for h in H:
     bh=move(ed.batch([i],(h,),'short','P1'),'cuda')
     if not bh['world_valid'].all():continue
     bh['external_no_robot']=True;Y=bh['world'];p=m.forward_mean(bh)['mean'];assert torch.isfinite(p).all();expanded.append(dict(index=i,parent=r['parent_component'],h=h,method='no_traj',model=float((p.double()-Y.double()).square().mean()),copy=float(Y.double().square().mean()),cosine=float(torch.nn.functional.cosine_similarity(p.flatten(),Y.flatten(),dim=0))))
    if i%50==0:print(group,'expanded',i,flush=True)
  atomic(run/'evaluation.json',dict(records=records,actions=actions,expanded=expanded,checkpoint_sha256=sha(ck),brt=brt()));report(f'## {group} 评估完成\n固定48窗/6parent，{len(records)}世界记录；扩展{len(expanded)}时域记录。')
 finally:guard.finish()

def finish():
 assert not (CONTROL/'FINAL_RESULTS.json').exists()
 ds={g:json.loads((T/f'results/e103d_{g.lower()}_s90_20261115_02/evaluation.json').read_text()) for g in ['D0','D1']};out={};lines=[]
 rng=np.random.default_rng(90)
 def table(key):
  result=[]
  for h in H:
   methods=['no_traj'] if key=='expanded' else ['gt_action','no_traj','pred_N1','pred_N8']
   for method in methods:
    rec={g:[r for r in ds[g][key] if r['h']==h and r['method']==method] for g in ds};parents=sorted({r['parent'] for r in rec['D0']});assert parents==sorted({r['parent'] for r in rec['D1']});a={g:np.array([[np.mean([r[k] for r in rec[g] if r['parent']==p]) for k in ['model','copy','cosine']] for p in parents]) for g in ds};assert np.allclose(a['D0'][:,1],a['D1'][:,1]);ix=np.random.default_rng(90).integers(len(parents),size=(2000,len(parents)));delta=(a['D0'][:,0].mean()-a['D1'][:,0].mean())/a['D0'][:,1].mean();boot=(a['D0'][ix,0].mean(1)-a['D1'][ix,0].mean(1))/a['D0'][ix,1].mean(1)
    row=dict(h=h,method=method,windows=len(rec['D0']),parents=len(parents),delta_skill_pp=float(delta*100),delta_ci95_pp=(np.quantile(boot,[.025,.975])*100).tolist())
    for g in ds:
     skill=1-a[g][:,0].mean()/a[g][:,1].mean();bskill=1-a[g][ix,0].mean(1)/a[g][ix,1].mean(1);row[g]=dict(skill_percent=float(skill*100),ci95_percent=(np.quantile(bskill,[.025,.975])*100).tolist(),cosine=float(a[g][:,2].mean()))
    result.append(row)
  return result
 out['fixed']=table('records');out['expanded']=table('expanded');act=[]
 for method in ['N1','N8_mean']:
  for h in [8,50]:
   for metric in ['joint_L1_rad','joint_RMSE_rad','gripper_L1_mm','gripper_RMSE_mm']:
    parents=sorted({r['parent'] for r in ds['D0']['actions']});vectors={g:np.array([np.mean([r[metric] for r in ds[g]['actions'] if r['parent']==p and r['h']==h and r['method']==method]) for p in parents]) for g in ds};vals={g:float(v.mean()) for g,v in vectors.items()};ix=np.random.default_rng(90).integers(len(parents),size=(2000,len(parents)));boot={g:v[ix].mean(1) for g,v in vectors.items()};act.append(dict(method=method,h=h,metric=metric,**vals,D0_ci95=np.quantile(boot['D0'],[.025,.975]).tolist(),D1_ci95=np.quantile(boot['D1'],[.025,.975]).tolist(),relative_change_percent=(vals['D1']/vals['D0']-1)*100,relative_change_ci95_percent=np.quantile((boot['D1']/boot['D0']-1)*100,[.025,.975]).tolist()))
 out['action']=act;nt=[r for r in out['fixed'] if r['method']=='no_traj' and r['h'] in [25,50,100]];gt=[r for r in out['fixed'] if r['method']=='gt_action' and r['h'] in [25,50,100]];ac=[r for r in act if r['method']=='N1' and 'L1' in r['metric']];passed=sum(r['delta_ci95_pp'][0]>0 for r in nt)>=2 and all(r['delta_ci95_pp'][0]>-1 for r in gt) and all(r['relative_change_percent']<=5 for r in ac);out['decision']=dict(external_data_effective=passed,verdict='外部无动作数据有效' if passed else '无增益',no_traj_improved_horizons=sum(r['delta_ci95_pp'][0]>0 for r in nt),gt_action_noninferiority=all(r['delta_ci95_pp'][0]>-1 for r in gt),action_within5pct=all(r['relative_change_percent']<=5 for r in ac),bootstrap=2000,seed=90,single_seed_limitation=True)
 for key,title in [('fixed','固定48窗/6 parent'),('expanded','E93扩展 no_traj')]:
  lines += ['## '+title,'|h|条件|窗/parent|D0 skill% [CI]|D1 skill% [CI]|D1−D0 pp [CI]|','|---:|---|---|---|---|---|']
  for r in out[key]:
   fmt=lambda d:f"{d['skill_percent']:.4f} [{d['ci95_percent'][0]:.4f},{d['ci95_percent'][1]:.4f}]";ci=r['delta_ci95_pp'];lines.append(f"|{r['h']}|{r['method']}|{r['windows']}/{r['parents']}|{fmt(r['D0'])}|{fmt(r['D1'])}|{r['delta_skill_pp']:.4f} [{ci[0]:.4f},{ci[1]:.4f}]|")
 lines+=['## 动作误差（parent等权）','|采样|h|指标|D0 [CI]|D1 [CI]|相对变化% [CI]|','|---|---:|---|---:|---:|---:|']
 for r in act:lines.append(f"|{r['method']}|{r['h']}|{r['metric']}|{r['D0']:.6f} [{r['D0_ci95'][0]:.6f},{r['D0_ci95'][1]:.6f}]|{r['D1']:.6f} [{r['D1_ci95'][0]:.6f},{r['D1_ci95'][1]:.6f}]|{r['relative_change_percent']:.3f} [{r['relative_change_ci95_percent'][0]:.3f},{r['relative_change_ci95_percent'][1]:.3f}]|")
 lines+=['## 预注册判定','```json',json.dumps(out['decision'],ensure_ascii=False),'```','指标先窗内聚合、再parent等权；parent配对bootstrap2000，seed90；单seed不估计seed间不确定性。所有checkpoint均固定终点，未按dev选型。']
 out['gpu_seconds']=used_seconds();out['gpu_hours']=used_seconds()/3600;atomic(CONTROL/'FINAL_RESULTS.json',out);report('\n'.join(lines));print(json.dumps(out['decision'],ensure_ascii=False),flush=True)
if __name__=='__main__':
 import sys
 try:
  if sys.argv[1]=='finish':finish()
  else:evaluate(sys.argv[1])
 except BaseException:atomic(CONTROL/('eval_error_'+sys.argv[1]+'_02.json'),dict(error=traceback.format_exc(),brt=brt()));raise
