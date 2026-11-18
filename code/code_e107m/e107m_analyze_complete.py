import os,sys,json,pickle,hashlib,numpy as np
from pathlib import Path
from e103d_runtime import *
import e107p_metrics as ep
from e104p_metrics import worlds
from e102a_finalize import action_summary
P=T/'results/e107p_diagnosis_20261116_02'
def loadj(p):return json.loads(p.read_text())
def stat(v,par):
 w0,w,ps=ep.weights(np.array(par));return ep.estimate(np.array(v),w0,w)
def ratio(a,b,par,root=False):
 w0,w,ps=ep.weights(np.array(par));a=np.array(a);b=np.array(b);x=(w0@a)/(w0@b);z=(w@a)/(w@b)
 if root:x=np.sqrt(x);z=np.sqrt(z)
 return dict(value=float(x),ci95=ep.interval(z))
def main():
 lock=loadj(CONTROL/'PRETRAIN_LOCK.json');entries=lock['entries'];em={e['index']:e for e in entries};summary={};prob={};chain={};vec={}
 lines=['## 固定终点：E106-C同口径技能与动作','主集211窗/27parent，+100为208窗/26parent；附48窗/6parent；parent等权2000次seed90。skill是标准化特征MSE相对复制改善，不是任务成功率。','|集|组/读法|h|GT / pred / no_traj / oracle skill %|','|---|---|---:|---|']
 for g in ('BASE','M0','M1'):
  art=CONTROL/('eval_'+g);x=loadj(art/'evaluation.json');prob[g]=loadj(art/'probabilistic.json');chain[g]=loadj(art/'chains.json');vec[g]=np.load(art/'vectors.npz');summary[g]={}
  for subset in ('all27','original48'):
   keep=lambda r:subset=='all27' or em[r['index']]['original48'];rr=[r for r in x['world'] if keep(r)];ar=[r for r in x['actions'] if keep(r)];wr=worlds(rr);ac=action_summary(ar);zero=[dict(r,model=next(t['zero_mse'] for t in prob[g] if (t['index'],t['h'],t['method'])==(r['index'],r['h'],r['method']))) for r in rr];zr=worlds(zero)
   summary[g][subset]=dict(mean_world=wr,zero_world=zr,actions=ac)
   for reading,data in [('mean',wr)]+([('z0',zr)] if g=='M1' else []):
    for h,v in data.items():lines.append(f"|{subset}|{g}/{reading}|{h}|"+'/'.join(f"{100*v['skill'][m]:.3f}" for m in ('gt_action','pred_N1','no_traj','oracle'))+'|')
 atomic(CONTROL/'SKILL_ACTION_SUMMARY.json',summary);report('\n'.join(lines))
 # Proper scores and paired bootstrap: same parent weights, no flattening across parents.
 scores=[];paired=[]
 for subset in ('all27','original48'):
  for h in (10,25,50,100):
   for cond in ('no_traj','gt_action','pred_N1','oracle'):
    rows={g:[r for r in prob[g] if r['h']==h and r['method']==cond and (subset=='all27' or em[r['index']]['original48'])] for g in prob};par=[r['parent'] for r in rows['M0']]
    assert all([r['index'] for r in v]==[r['index'] for r in rows['M0']] for v in rows.values())
    get=lambda g,k:np.array([r[k] for r in rows[g]])
    for g in rows:
     item=dict(subset=subset,h=h,condition=cond,group=g,windows=len(par),parents=len(set(par)))
     for k in ('energy','crps','spread2','coverage'):item[k]=stat(get(g,k),par)
     item['spread_skill']=ratio(get(g,'spread2'),get(g,'mean_mse'),par,True)
     item['best_of_K_skill']=ratio(get(g,'copy')-get(g,'best_mse'),get(g,'copy'),par)
     scores.append(item)
    d=dict(subset=subset,h=h,condition=cond,windows=len(par),parents=len(set(par)))
    for k in ('energy','crps'):d[k+'_improvement']=stat(get('M0',k)-get('M1',k),par)
    d['mean_skill_gain']=ratio(get('M0','mean_mse')-get('M1','mean_mse'),get('M0','copy'),par);d['z0_skill_gain']=ratio(get('M0','mean_mse')-get('M1','zero_mse'),get('M0','copy'),par);paired.append(d)
 atomic(CONTROL/'PROPER_SCORES.json',scores);atomic(CONTROL/'PAIRED.json',paired)
 # E100-C endpoint-only open-loop comparison, no true middle state input.
 chainout=[];chainpaired=[]
 for subset in ('all27','original48'):
  for k in (25,50):
   rows={g:[r for r in chain[g] if r['k']==k and (subset=='all27' or em[r['index']]['original48'])] for g in chain};par=[r['parent'] for r in rows['M0']]
   for g in rows:
    o=dict(subset=subset,k=k,group=g,windows=len(par),parents=len(set(par)))
    for key in ('mean_rollout_mse','sample_rollout_mse','sample_ensemble_endpoint_mse'):o[key]=stat([r[key] for r in rows[g]],par)
    chainout.append(o)
   base=np.array([r['mean_rollout_mse'] for r in rows['M0']]);o=dict(subset=subset,k=k,windows=len(par),parents=len(set(par)))
   for key in ('mean_rollout_mse','sample_rollout_mse','sample_ensemble_endpoint_mse'):o[key+'_improvement']=stat(base-np.array([r[key] for r in rows['M1']]),par)
   chainpaired.append(o)
 atomic(CONTROL/'OPENLOOP.json',dict(metrics=chainout,paired=chainpaired,contract='E100C open_second; zero history, masked state, reentry predicted features; original <=t plan only'))
 # Reuse exact E107-P projection, frozen gripper probe, mixed-start proxies and distribution functions.
 proj=np.load(P/'projection.npz');mu=proj['mean'];basis=proj['basis'];band=float(proj['bandwidth']);ridge=float(proj['ridge'])
 with (P/'grip_probe.pkl').open('rb') as f:probe=pickle.load(f)
 neighbors={r['index']:r for r in loadj(P/'neighbors.json')['rows']};numeric={r['index']:r for r in loadj(P/'dev_numeric.json')};curves=[];distributions=[];branches=[]
 gp=lambda a:np.stack([m.predict_proba(sc.transform(a))[:,1] for sc,m in probe],-1)
 for subset in ('all27','original48'):
  for h in (10,25,50,100):
   ee=[e for e in entries if h in e['horizons'] and (subset=='all27' or e['original48'])];ids=[e['index'] for e in ee];par=np.array([e['parent'] for e in ee]);w0,w,ps=ep.weights(par)
   pool=lambda a:a.reshape(len(a),3,-1,768).mean(2).reshape(len(a),2304)
   Y=np.stack([vec['BASE'][f'{i}_{h}_Y'] for i in ids]);C=np.stack([vec['BASE'][f'{i}_{h}_C'] for i in ids]);yy=pool(Y).astype('f8');yz=(yy-mu)@basis
   noise=yy+np.random.default_rng(90+h).normal(size=yy.shape)*.05*proj['train_std'];nz=(noise-mu)@basis
   for cond in ('no_traj','gt_action'):
    for g in ('BASE','M0','M1'):
     for reading in ('mean','zero') if g=='M1' else ('mean',):
      A=np.stack([vec[g][f'{i}_{h}_{cond}_{reading}'] for i in ids]);pp=pool(A).astype('f8');curve=ep.summarize_curve(Y,C,A,par);curve.update(subset=subset,h=h,condition=cond,group=g,reading=reading);curves.append(curve)
      dist=ep.distribution(yz,(pp-mu)@basis,nz,par,band,ridge,w0,w);dist.update(subset=subset,h=h,condition=cond,group=g,reading=reading);distributions.append(dist)
     if g=='M1':
      ss=np.stack([vec[g][f'{i}_{h}_{cond}_samples_pool'] for i in ids]);assert ss.shape==(len(ids),4,2304)
      raws=np.stack([vec[g][f'{i}_{h}_{cond}_samples_raw'] for i in ids]);raws=raws.reshape(len(ids),4,*Y.shape[1:])
      curve=ep.summarize_curve(np.repeat(Y,4,axis=0),np.repeat(C,4,axis=0),raws.reshape(len(ids)*4,*Y.shape[1:]),np.repeat(par,4));curve.update(subset=subset,h=h,condition=cond,group=g,reading='sample_population');curves.append(curve)
      # Each sample member is a full one-per-window draw; preserve E107P support definition.
      for j in range(4):
       dist=ep.distribution(yz,(ss[:,j]-mu)@basis,nz,par,band,ridge,w0,w);dist.update(subset=subset,h=h,condition=cond,group=g,reading='sample'+str(j));distributions.append(dist)
      probs=gp(ss.reshape(-1,2304)).reshape(len(ids),4,2);hi=(10,25,50).index(h) if h!=100 else None
      mixed=np.array([hi is not None and neighbors.get(i,{}).get('candidate',False) and neighbors.get(i,{}).get('eligible',False) and neighbors.get(i,{}).get('mixed',[False]*3)[hi] for i in ids])
      trueprob=gp(yy)
      for tag,mask in [('all',np.ones(len(ids),bool)),('mixed',mixed)]:
       o=dict(subset=subset,h=h,condition=cond,group=tag,windows=int(mask.sum()),parents=len(set(par[mask])),insufficient=bool(mask.sum()<30 or len(set(par[mask]))<6),per_gripper=[])
       if mask.any():
        for k in range(2):
         pp=probs[mask,:,k];measured=np.array([numeric[i]['future'][hi][k] if hi is not None else -1 for i in np.array(ids)[mask]])
         both=((pp<.3).any(1)&(pp>.7).any(1)).astype(float);predopen=(pp>.5).mean(1);truth=(measured>0).astype(float)
         o['per_gripper'].append(dict(gripper=k,two_cluster_proxy=stat(both,par[mask]),sample_open_probability=stat(pp.mean(1),par[mask]),sample_open_fraction=stat(predopen,par[mask]),true_open_fraction=stat(truth,par[mask]) if hi is not None else None,open_fraction_gap=stat(predopen-truth,par[mask]) if hi is not None else None,histogram_counts=np.histogram(pp,bins=np.linspace(0,1,11))[0].tolist(),per_window_probabilities=pp.tolist()))
       branches.append(o)
    print('distribution',subset,h,cond,flush=True);atomic(CONTROL/'distribution_partial.json',distributions);atomic(CONTROL/'curves_partial.json',curves)
    report('## 分布指标完成 '+subset+' +'+str(h)+' '+cond+'\n复用E107P原脚本函数与训练PCA32、核带宽、ridge、探针、混合起点代理；四个样本成员分别作一窗一抽样分布，避免重复真实窗破坏kNN支持半径。\n')
 atomic(CONTROL/'DISTRIBUTION.json',distributions);atomic(CONTROL/'SHRINKAGE.json',curves);atomic(CONTROL/'BRANCH.json',branches)
 primary=[r for r in paired if r['subset']=='all27' and r['condition']=='no_traj'];sp=[r for r in scores if r['subset']=='all27' and r['condition']=='no_traj' and r['group']=='M1'];cc=[r for r in chainpaired if r['subset']=='all27']
 proper=all(sum(r[k+'_improvement']['ci95'][0]>0 for r in primary)>=2 for k in ('energy','crps'));noninfer=all(r['mean_skill_gain']['ci95'][0]>-.01 for r in primary);spread=all(.8<=r['spread_skill']['value']<=1.2 for r in sp);openok=all(r['sample_rollout_mse_improvement']['ci95'][0]>0 for r in cc)
 decision=dict(multimodal_effective=proper and noninfer and spread and openok,proper=proper,noninfer=noninfer,spread=spread,openloop=openok,collapsed_horizons=[r['h'] for r in sp if r['spread_skill']['value']<.5],criterion=lock['preregistered'],default_unchanged=True,M2=False,brt=brt())
 atomic(CONTROL/'DECISION.json',decision)
 lines=['## 概率指标与配对判定','正改善= M0分数−M1分数；能量/CRPS越低越好。CRPS为逐坐标一维边际分数；能量用完整时域特征向量的RMS距离。coverage仅逐坐标min/max覆盖率，不是联合覆盖概率。','|h|ES改善 [CI]|CRPS改善 [CI]|均值skill增益pp [CI]|spread/skill|坐标覆盖率|','|---:|---|---|---|---|---|']
 fmt=lambda a:f"{a['value']:.6f} [{a['ci95'][0]:.6f},{a['ci95'][1]:.6f}]"
 for r,s in zip(primary,sp):lines.append(f"|{r['h']}|{fmt(r['energy_improvement'])}|{fmt(r['crps_improvement'])}|"+fmt(dict(value=100*r['mean_skill_gain']['value'],ci95=np.array(r['mean_skill_gain']['ci95'])*100))+f"|{fmt(s['spread_skill'])}|{fmt(s['coverage'])}|")
 lines+=['','## 开环两跳','|k|样本推演MSE改善 [CI]|每跳样本均值推演改善 [CI]|样本端点集成改善 [CI]|','|---:|---|---|---|']
 for r in cc:lines.append(f"|{r['k']}|{fmt(r['sample_rollout_mse_improvement'])}|{fmt(r['mean_rollout_mse_improvement'])}|{fmt(r['sample_ensemble_endpoint_mse_improvement'])}|")
 lines+=['','## 预注册结论',json.dumps(decision,ensure_ascii=False),'满足全部判据。' if decision['multimodal_effective'] else '当前设定下多峰建模无明确收益，不进入 M2，不改默认。','样本塌缩时域（spread/skill<0.5）：'+str(decision['collapsed_horizons'])+'。','分叉为训练近邻机械混合代理，不证明不可约多峰；K4两簇代理=同一窗同时出现p<.3和p>.7，直方图与样本开比例见BRANCH.json。混合格24/15、30/19、25/19窗/parent，+10/+50窗不足30，+100没有代理。','完整四条件/两集合的skill CI、动作逐维/逐步/事件、best-of-K、分布指标（含MMD/PR/C2ST/Fréchet/熵代理）、方差与分叉见本control JSON。仅seed90；不估计seed间不确定性，不作部署结论。']
 report('\n'.join(lines))
 bytecount=sum(p.stat().st_size for root in (ROOT,CONTROL,T/'results/e107m_m0_s90_20261116_01',T/'results/e107m_m1_s90_20261116_01') for p in root.rglob('*') if p.is_file());assert bytecount<10_000_000_000;assert sha(Path(lock['checkpoints']['BASE']))==lock['resume_sha']
 audit=dict(status='complete',GPUh_upper=used_seconds()/3600,new_bytes=bytecount,old_checkpoint_unchanged=True,brt=brt());atomic(CONTROL/'FINAL_AUDIT.json',audit);report('## 最终资源审计\n'+json.dumps(audit,ensure_ascii=False)+'\n必需工作已完成；无需用户追加决定，默认不变。')
 with (T/'docs/experiment_log/EXPERIMENT_LOG.md').open('a') as f:
  __import__('fcntl').flock(f,__import__('fcntl').LOCK_EX);f.write('\n## E107-M | '+brt()+' BRT\n目标：世界头低成本多峰验证，仅M0/M1；改动：code_e107m，32D z/256隐藏/K4能量+0.001均值MSE，旧头与主干冻结。命令：setsid nohup env PYTHONDONTWRITEBYTECODE=1 '+str(Path('/mnt/workspace/dengjianfeng/venvs/e29/bin/python'))+' -u e107m_pipeline.py。目录：results/e107m_*_20261116_01；报告：notes/e107m_multimodal_head_20261116.md。结果：'+json.dumps(decision,ensure_ascii=False)+'；资源：'+json.dumps(audit,ensure_ascii=False)+'；结论：'+('满足判据；本轮仍不做M2。' if decision['multimodal_effective'] else '当前设定下多峰建模无明确收益，不进入M2，不改默认。')+' 状态：COMPLETE。\n')
 with (T/'notes/worklog.md').open('a') as f:
  __import__('fcntl').flock(f,__import__('fcntl').LOCK_EX);f.write('\n- '+brt()+' BRT E107-M COMPLETE：M0/M1固定3000步，主27parent/附48窗，结论='+str(decision['multimodal_effective'])+'，GPUh='+str(audit['GPUh_upper'])+'，报告 notes/e107m_multimodal_head_20261116.md。\n')
if __name__=='__main__':main()
