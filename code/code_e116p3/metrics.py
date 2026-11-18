import os,sys,json,pickle,glob
from pathlib import Path
import numpy as np
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=T/'results/e116p3_20261117_01';OLD=T/'results/e99b3_20261115_01';CF=T/'results/e102s2_cf_20261116_01';E102=T/'results/e102s_20261115_01';VLM=T/'results/e113v_vlm_multiframe_20261116_01';HUM=T/'results/e114h_human_stage_20261116/boundaries';REPORT=T/'notes/e116p3_language_alignment_2026-11-17.md'
S=[f'S{i:02d}' for i in range(1,11)];P=[f'P{i:03d}' for i in [11,21,22,31,32,41,42,51,52,61,62,71,72,73,81,82,91,92,101,102,103]];C=['cam_high','cam_left_wrist','cam_right_wrist']
def fin(x):return float(x) if np.isfinite(x) else None
def auc(y,x,w=None):
 y=np.asarray(y,int);x=np.asarray(x,float);w=np.ones(len(y)) if w is None else np.asarray(w,float);ok=np.isfinite(x)&(w>0);y=y[ok];x=x[ok];w=w[ok]
 if not np.any(y==1) or not np.any(y==0):return np.nan
 o=np.argsort(x,kind='stable');y=y[o];x=x[o];w=w[o];st=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1];wp=np.add.reduceat(w*(y==1),st);wn=np.add.reduceat(w*(y==0),st);return float(np.sum(wp*(np.cumsum(wn)-.5*wn))/(wp.sum()*wn.sum()))
def ci(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)];return np.quantile(a,[.025,.975]).tolist() if len(a) else [None,None]
def cm_stats(y,p):
 cm=np.zeros((10,10),int);np.add.at(cm,(y,p),1);tp=np.diag(cm);d=cm.sum(0)+cm.sum(1);return dict(top1=float(np.trace(cm)/max(cm.sum(),1)),macro_f1=float(np.mean(np.divide(2*tp,d,out=np.zeros(10,float),where=d>0))),confusion=cm.tolist())
def top3(y,x):
 if len(y)==0:return np.nan
 idx=np.argpartition(x,-3,axis=1)[:,-3:];return float(np.mean(np.any(idx==np.asarray(y)[:,None],axis=1)))
def logit(p):p=np.clip(np.asarray(p,float),1e-7,1-1e-7);return np.log(p/(1-p))
def main():
 assert not (R/'metrics.json').exists()
 lock=json.loads((R/'text_lock.json').read_text());srcs=json.loads((OLD/'sources.json').read_text());src={x['parent']:x for x in srcs if x['pool']=='dev' and x['cid']!='piper_4dc107335389f5711a0f65e0'};parents=sorted(src);pari={p:i for i,p in enumerate(parents)};rng=np.random.default_rng(90);bdraw=rng.integers(20,size=(2000,20));
 sims={};fm={};segments={p:[] for p in parents}
 for p in parents:
  z=np.load(R/(p+'_similarities.npz'));sims[p]=z['prob'].astype('f4');fm[p]={int(f):i for i,f in enumerate(z['frames'])}
 for z in json.loads((OLD/'stages_dev.json').read_text()):
  if z['parent_id'] in segments:segments[z['parent_id']].append(z)
 truth={}
 for p in parents:
  for z in segments[p]:
   for f in range(int(z['stage_start']),int(z['stage_end_exclusive'])):truth[p,f]=z['stage_id']
 def stlog(p,f,g='start_plus_end',l='en',fusion='mean'):
  pr=sims[p][fm[p][int(f)],:,:60].astype('f4');z=logit(pr).reshape(3,10,6);j={'name':0,'start':1,'start_plus_end':2}[g]+(3 if l=='zh' else 0);return z[0,:,j] if fusion=='high' else z[:,:,j].mean(0)
 fs={p:np.arange(0,int(src[p]['n']),30,dtype=int) for p in parents};sens={};per={}
 for g in ['name','start','start_plus_end']:
  for l in ['en','zh']:
   for fusion in ['high','mean']:
    rows=[]
    for p in parents:
     f=np.array([q for q in fs[p] if (p,int(q)) in truth and int(q) in fm[p]],int);y=np.array([S.index(truth[p,int(q)]) for q in f],int);x=np.array([stlog(p,q,g,l,fusion) for q in f]);rows.append((y,x.argmax(1),x))
    per[g,l,fusion]=rows
 def stage_metric(rows,boot=False):
  ys=np.concatenate([r[0] for r in rows]);xs=np.concatenate([r[2] for r in rows]);ps=xs.argmax(1);v=cm_stats(ys,ps);hits=np.any(np.argpartition(xs,-3,axis=1)[:,-3:]==ys[:,None],axis=1);v['top3']=float(hits.mean())
  if boot:
   pidx=np.concatenate([np.full(len(r[0]),i,int) for i,r in enumerate(rows)]);by=np.zeros((20,100),float);np.add.at(by,(pidx,ys*10+ps),1);bc=(bdraw@by).reshape(-1,10,10);tp=np.diagonal(bc,axis1=1,axis2=2);den=bc.sum(1)+bc.sum(2);mf=np.mean(np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0),axis=1);acc=tp.sum(1)/np.maximum(bc.sum((1,2)),1);bh=np.bincount(pidx,weights=hits.astype(float),minlength=20);bn=np.bincount(pidx,minlength=20);t3=(bdraw@bh)/np.maximum(bdraw@bn,1);bootv=np.column_stack([acc,mf,t3]);v.update(top1_ci95=ci(acc),macro_f1_ci95=ci(mf),top3_ci95=ci(t3),_boot=bootv)
  return v
 mainrows=per['start_plus_end','en','mean'];highrows=per['start_plus_end','en','high'];zero=stage_metric(mainrows,True);high=stage_metric(highrows,True)
 # CF H0 trained causal predicate head -> calibrated stage emissions -> framewise argmax (no HMM).
 sys.path.insert(0,str(T/'code_e102s2'))
 # Read-only pickle import shim: the stored linear model's module imports SciPy's optimizer,
 # but inference uses only numpy and expit. No fitting/optimization is run.
 import types
 scipy=types.ModuleType('scipy');opt=types.ModuleType('scipy.optimize');special=types.ModuleType('scipy.special')
 opt.minimize=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('fit disabled'))
 special.expit=lambda x:1/(1+np.exp(-np.asarray(x)))
 scipy.optimize=opt;scipy.special=special;sys.modules['scipy']=scipy;sys.modules['scipy.optimize']=opt;sys.modules['scipy.special']=special
 with (CF/'H0_full.pkl').open('rb') as f:head=pickle.load(f)
 with (CF/'final_aux.pkl').open('rb') as f:aux=pickle.load(f)['H0']
 devlabels=json.loads((OLD/'dev_labels.json').read_text());allfr={p:sorted(set(fs[p].tolist()+[int(r['frame']) for r in devlabels if r['parent']==p and r['causal_valid']])) for p in parents};cfprob={}
 for p in parents:
  b=np.load(E102/'features'/p/'b.npy',mmap_mode='r');fr=np.array(allfr[p],int);x=head['sc'].transform(np.asarray(b[fr],dtype='f4')).astype('f4');q=np.stack([m.predict_proba(x)[:,1] if hasattr(m,'predict_proba') else np.full(len(x),float(m)) for m in head['head']],1).astype('f4');z=np.log(np.clip(q,1e-5,1-1e-5)/np.clip(1-q,1e-5,1));cal=aux['cal'];qcal=1/(1+np.exp(-np.clip(z*cal[:,0]+cal[:,1],-40,40)));qcal[:,P.index('P073')]=.5;e=np.c_[qcal,np.ones(len(qcal))]@aux['w']-np.log(np.maximum(aux['prior'],1e-30));assert np.isfinite(e).all();cfprob[p]={int(f):(e[i],q[i]) for i,f in enumerate(fr)}
 cfrows=[]
 for i,p in enumerate(parents):
  y,x=mainrows[i][0],mainrows[i][2];f=np.array([q for q in fs[p] if (p,int(q)) in truth and int(q) in fm[p]],int);e=np.array([cfprob[p][int(q)][0] for q in f]);cfrows.append((y,e.argmax(1),e))
 cf=stage_metric(cfrows,True);delta=zero['_boot']-cf['_boot'];stage_delta=dict(top1=zero['top1']-cf['top1'],top1_ci95=ci(delta[:,0]),macro_f1=zero['macro_f1']-cf['macro_f1'],macro_f1_ci95=ci(delta[:,1]));zero.pop('_boot');high.pop('_boot');cf.pop('_boot')
 for k,rows in per.items():sens['/'.join(k)]=stage_metric(rows,False)
 # Human marker labels on the same 30-frame sample.
 humanfiles=sorted(glob.glob(str(HUM/'*.json')));human=[];humandata={}
 for path in humanfiles:
  h=json.loads(Path(path).read_text());p=h['parent'];humandata[p]=h
  if p not in parents:continue
  f=np.array([q for q in fs[p] if (p,int(q)) in truth and int(q) in fm[p]],int);x=np.array([stlog(p,q) for q in f]);bs=sorted(h['boundaries'],key=lambda z:z['frame']);yy=[]
  for q in f:
   prev=[b for b in bs if int(b['frame'])<=q];yy.append(S.index(prev[-1]['stage']) if prev else -1)
  mask=np.array(yy)>=0
  if mask.any():
   m=cm_stats(np.array(yy)[mask],x[mask].argmax(1));human.append(dict(parent=p,n=int(mask.sum()),top1=m['top1'],macro_f1=m['macro_f1'],confusion=m['confusion']))
 # Predicate rows and parent-bootstrap macro AUROC.
 spec=json.loads((OLD/'spec.json').read_text());pcam={z['id']:C.index(z['primary_camera']) for z in spec['predicates']};pdraw=np.random.default_rng(10201).integers(20,size=(2000,20));pweights=np.stack([np.bincount(x,minlength=20) for x in pdraw]);baseline={m:{} for m in ['VLM_zero_shot','VLM_multiframe_V3','SigLIP2_linear_b']}
 for p in parents:
  for name,path,key in [('VLM_zero_shot',OLD/(p+'_vlm.npz'),'prob'),('SigLIP2_linear_b',OLD/(p+'_scores.npz'),'b_prob')]:
   if path.exists():
    z=np.load(path)
    for i,f in enumerate(z['frames']):
     for j,pid in enumerate(P):baseline[name][p,int(f),pid]=float(z[key][i,j])
  path=VLM/(p+'_dev_V3.npz')
  if path.exists():
   z=np.load(path)
   for i,f in enumerate(z['frames']):
    for j,pid in enumerate(P):baseline['VLM_multiframe_V3'][p,int(f),pid]=float(z['prob'][i,j])
 labels=[r for r in devlabels if r['causal_valid'] and r['causal_answer'] in ('yes','no') and r['parent'] in parents and r['predicate']!='P073'];ys=[];px=[];jx=[];scores={k:[] for k in ['SigLIP2_text_primary','SigLIP2_text_cam_mean','SigLIP2_text_cam_max','SigLIP2_text_zh_primary','CF_H0','VLM_zero_shot','VLM_multiframe_V3','SigLIP2_linear_b']}
 for r in labels:
  p=r['parent'];f=int(r['frame']);pid=r['predicate'];j=P.index(pid)
  if f not in fm[p]:continue
  a=sims[p][fm[p][f],:,60+2*j].astype(float);z=sims[p][fm[p][f],:,61+2*j].astype(float);vals={'SigLIP2_text_primary':logit(a[pcam[pid]]),'SigLIP2_text_cam_mean':logit(a).mean(),'SigLIP2_text_cam_max':logit(a).max(),'SigLIP2_text_zh_primary':logit(z[pcam[pid]]),'CF_H0':cfprob[p][f][1][j]}
  for name in baseline:vals[name]=baseline[name].get((p,f,pid),np.nan)
  ys.append(int(r['causal_answer']=='yes'));px.append(pari[p]);jx.append(j)
  for k in scores:scores[k].append(vals[k])
 y=np.array(ys,int);px=np.array(px,int);jx=np.array(jx,int);scores={k:np.array(v,float) for k,v in scores.items()}
 def pred_auc(mask,score,weights=None):
  vals=[]
  for j,pid in enumerate(P):
   if pid=='P073':continue
   m=mask&(jx==j)&np.isfinite(score);vals.append(auc(y[m],score[m],None if weights is None else weights[m]))
  return np.array(vals)
 def auc_boot(yv,sv,pv):
  yv=np.asarray(yv,int);sv=np.asarray(sv,float);pv=np.asarray(pv,int)
  if not np.any(yv==1) or not np.any(yv==0):return np.full(len(pweights),np.nan)
  order=np.argsort(sv,kind='stable');yv=yv[order];sv=sv[order];pv=pv[order];starts=np.r_[0,np.flatnonzero(np.diff(sv)!=0)+1]
  wm=pweights[:,pv];wp=np.add.reduceat(wm*(yv[None,:]==1),starts,axis=1);wn=np.add.reduceat(wm*(yv[None,:]==0),starts,axis=1);num=(wp*(np.cumsum(wn,axis=1)-.5*wn)).sum(axis=1);return np.divide(num,wp.sum(1)*wn.sum(1),out=np.full(len(pweights),np.nan),where=(wp.sum(1)*wn.sum(1))>0)
 def macro(mask,score):
  ppids=[pid for pid in P if pid!='P073'];vals=[];boot=[]
  for j,pid in enumerate(P):
   if pid=='P073':continue
   m=mask&(jx==j)&np.isfinite(score);vals.append(auc(y[m],score[m]));boot.append(auc_boot(y[m],score[m],px[m]))
  vv=np.asarray(vals);bbmat=np.stack(boot,axis=1);pt=float(np.nanmean(vv)) if np.isfinite(vv).any() else np.nan;bb=np.nanmean(bbmat,axis=1)
  return pt,vv,bb
 pres={}
 for k,x in scores.items():
  m=np.isfinite(x);pt,pv,bb=macro(m,x);pres[k]=dict(n=int(m.sum()),parents=len(set(px[m])),macro_auroc=fin(pt),ci95=ci(bb),estimable=int(np.isfinite(pv).sum()),per_predicate={pid:fin(pv[i]) for i,pid in enumerate([q for q in P if q!='P073'])})
 paired={};primary=scores['SigLIP2_text_primary']
 for k,ref in [('VLM_zero_shot',.629),('VLM_multiframe_V3',.683),('SigLIP2_linear_b',.795),('CF_H0',.908)]:
  m=np.isfinite(primary)&np.isfinite(scores[k]);a,_,ab=macro(m,primary);b,_,bb=macro(m,scores[k]);paired[k]=dict(n=int(m.sum()),parents=len(set(px[m])),primary_on_common=fin(a),comparator_on_common=fin(b),difference=fin(a-b),ci95=ci(ab-bb),published_reference=ref)
 table=[]
 for j,pid in enumerate(P):
  if pid!='P073':table.append(dict(predicate=pid,**{k:pres[k]['per_predicate'].get(pid) for k in ['SigLIP2_text_primary','SigLIP2_text_cam_mean','SigLIP2_text_cam_max','SigLIP2_text_zh_primary','CF_H0']}))
 # Within-parent time AUROC and 3 representative human-labeled curves (SVG, dependency-free).
 tstage={s:[] for s in S};tparent={};curve=[];win={off:[] for off in [-90,-60,-30,0,30,60,90]}
 for p in parents:
  fr=np.array([q for q in fs[p] if (p,int(q)) in truth and int(q) in fm[p]],int)
  if not len(fr):continue
  yy=np.array([S.index(truth[p,int(q)]) for q in fr]);xx=np.array([stlog(p,q) for q in fr]);vals={}
  for j,sid in enumerate(S):
   a=auc((yy==j).astype(int),xx[:,j]);vals[sid]=fin(a)
   if np.isfinite(a):tstage[sid].append(a)
  tparent[p]=vals
  if p in humandata and len(curve)<3:curve.append((p,fr,xx,humandata[p]))
  for z in segments[p]:
   sid=z['stage_id'];start=int(z['stage_start'])
   if start<=0:continue
   for off in win:
    f0=start+off;q=int(fr[np.argmin(abs(fr-f0))])
    if abs(q-f0)<=16:win[off].append(float(stlog(p,q)[S.index(sid)]))
 cols=['#d62728','#1f77b4','#2ca02c','#9467bd','#ff7f0e','#8c564b','#17becf','#e377c2','#7f7f7f','#bcbd22'];svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="900" viewBox="0 0 1100 900"><rect width="100%" height="100%" fill="white"/><text x="550" y="28" text-anchor="middle">SigLIP2 stage-text similarity with human boundary markers</text>']
 for panel,(p,fr,x,h) in enumerate(curve):
  top=50+panel*280;left=70;w=990;hh=200;mx=max(int(fr[-1]),1);svg.append(f'<text x="{left}" y="{top+18}">{p} ({len(fr)} 30-frame samples)</text>')
  for tick in range(6):
   yy=top+38+tick*hh/5;svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+w}" y2="{yy:.1f}" stroke="#ddd"/>')
  for b in h['boundaries']:
   xx=left+w*int(b['frame'])/mx;svg.append(f'<line x1="{xx:.1f}" y1="{top+38}" x2="{xx:.1f}" y2="{top+38+hh}" stroke="#777" stroke-dasharray="3 3"/>')
  prob=1/(1+np.exp(-np.clip(x,-35,35)))
  for j in range(10):
   pts=' '.join(f'{left+w*int(f)/mx:.1f},{top+38+(1-prob[i,j])*hh:.1f}' for i,f in enumerate(fr));svg.append(f'<polyline fill="none" stroke="{cols[j]}" stroke-width="1.4" points="{pts}"/>')
  svg.append(f'<text x="{left}" y="{top+260}" font-size="12">Dashed vertical lines: human stage starts; colors S01–S10 follow text_lock.json.</text>')
 svg.append('</svg>');(R/'stage_similarity_curves.svg').write_text('\n'.join(svg))
 hoverall=dict(clips=len(human),frames=sum(a['n'] for a in human),top1=float(np.average([a['top1'] for a in human],weights=[a['n'] for a in human])) if human else None,macro_f1=float(np.mean([a['macro_f1'] for a in human])) if human else None,per_parent=human)
 result=dict(protocol=dict(primary='start_plus_end English; mean three-camera logits',threshold_top1_ci_lower=.20,random_chance=.10,stage_bootstrap=dict(draws=2000,seed=90),predicate_bootstrap=dict(draws=2000,seed=10201),excluded='piper_4dc107335389f5711a0f65e0',dev_label_basis='model generated; development reference only',P073='fixed abstention'),stage=dict(main=zero,cam_high=high,CF_H0=cf,zero_minus_CF_H0=stage_delta,sensitivities=sens),human_stage=hoverall,predicate=dict(methods=pres,paired=paired,per_predicate=table),time_consistency=dict(per_stage={k:dict(parent_stage_pairs=len(v),mean_auroc=fin(np.mean(v)) if v else None) for k,v in tstage.items()},per_parent=tparent,boundary_window_mean_current_logit={str(k):fin(np.mean(v)) if v else None for k,v in win.items()}),files=['stage_similarity_curves.svg'])
 (R/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 fmt=lambda x:'NA' if x is None else f'{x:.3f}';lines=['\n## 4. 评估结果（单次冻结 dev；模型生成标签仅作开发参考）\n\n### 阶段零样本与监督对照\n',f"主结果 start+end 英文、三相机 logit 均值：top-1={zero['top1']:.4f} [{zero['top1_ci95'][0]:.4f},{zero['top1_ci95'][1]:.4f}]；top-3={zero['top3']:.4f} [{zero['top3_ci95'][0]:.4f},{zero['top3_ci95'][1]:.4f}]；macro-F1={zero['macro_f1']:.4f} [{zero['macro_f1_ci95'][0]:.4f},{zero['macro_f1_ci95'][1]:.4f}]。10 类随机 top-1=0.10；预注册通过门槛为 CI 下界 >0.20。\n",f"cam_high top-1={high['top1']:.4f} [{high['top1_ci95'][0]:.4f},{high['top1_ci95'][1]:.4f}]。CF H0 同帧逐帧 emission argmax（无 HMM）top-1={cf['top1']:.4f} [{cf['top1_ci95'][0]:.4f},{cf['top1_ci95'][1]:.4f}]，macro-F1={cf['macro_f1']:.4f}；零样本−CF H0 top-1 差={stage_delta['top1']:.4f} [{stage_delta['top1_ci95'][0]:.4f},{stage_delta['top1_ci95'][1]:.4f}]。\n",'\n|文本组|语言|融合|Top-1|Top-3|Macro-F1|\n|---|---|---|---:|---:|---:|\n']
 for k,v in sens.items():
  g,l,f=k.split('/');lines.append(f"|{g}|{l}|{f}|{fmt(v['top1'])}|{fmt(v['top3'])}|{fmt(v['macro_f1'])}|\n")
 lines+=['\n主结果混淆矩阵（行=参考、列=预测，S01–S10）：\n\n```text\n']+[ ' '.join(map(str,r))+'\n' for r in zero['confusion']]+['```\n',f"\n人工 4 片段以人工 marker 区间对照同一抽帧：帧加权 top-1={fmt(hoverall['top1'])}，片段 macro-F1 均值={fmt(hoverall['macro_f1'])}，共 {hoverall['frames']} 帧。\n",'\n### 谓词零样本与对照\n\n|方法|记录数|parent|可估谓词|Macro AUROC [95%CI]|\n|---|---:|---:|---:|---:|\n']
 for k,v in pres.items():lines.append(f"|{k}|{v['n']}|{v['parents']}|{v['estimable']}|{fmt(v['macro_auroc'])} [{fmt(v['ci95'][0])},{fmt(v['ci95'][1])}]|\n")
 lines+=['\n|谓词|英文 primary|三相机均值|三相机最大|中文 primary|CF H0|\n|---|---:|---:|---:|---:|---:|\n']
 for r in table:lines.append('|'+ '|'.join([r['predicate']]+[fmt(r[k]) for k in ['SigLIP2_text_primary','SigLIP2_text_cam_mean','SigLIP2_text_cam_max','SigLIP2_text_zh_primary','CF_H0']])+'|\n')
 lines+=['\n配对差=本轮英文 primary 文本−基线（仅共同可用标注行；seed10201 parent bootstrap）：\n\n|Comparator|共同记录|primary AUROC|对照 AUROC|配对差 [95%CI]|旧结果参考|\n|---|---:|---:|---:|---:|---:|\n']
 for k,v in paired.items():lines.append(f"|{k}|{v['n']}|{fmt(v['primary_on_common'])}|{fmt(v['comparator_on_common'])}|{fmt(v['difference'])} [{fmt(v['ci95'][0])},{fmt(v['ci95'][1])}]|{v['published_reference']:.3f}|\n")
 lines+=['\nP073 固定弃权并排除；宏 AUROC 对有 yes/no 的其余谓词取平均。\n\n### 时间一致性\n\n|阶段|parent-stage AUROC 可估数|平均 AUROC|\n|---|---:|---:|\n']
 for k,v in result['time_consistency']['per_stage'].items():lines.append(f"|{k}|{v['parent_stage_pairs']}|{fmt(v['mean_auroc'])}|\n")
 lines.append('\n边界相对帧当前阶段文本平均 logit：'+', '.join(f'{k}: {fmt(v)}' for k,v in result['time_consistency']['boundary_window_mean_current_logit'].items())+'。三个人工片段曲线见 `results/e116p3_20261117_01/stage_similarity_curves.svg`。\n')
 with REPORT.open('a') as f:f.write(''.join(lines))
 print(json.dumps(dict(stage_top1=zero['top1'],stage_ci=zero['top1_ci95'],CF_top1=cf['top1'],predicate={k:(v['macro_auroc'],v['ci95'],v['n']) for k,v in pres.items()},paired=paired,human=hoverall,bytes=sum(x.stat().st_size for x in R.iterdir() if x.is_file())),ensure_ascii=False))
if __name__=='__main__':main()
