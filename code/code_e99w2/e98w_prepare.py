from e98w_core import *
OUT=T/'results/e98w_control_20261114_01'
def put(name,x):
 p=OUT/name;p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not (OUT/'CROSSFIT_LOCK.json').exists()
 _,tr,_=datasets();st=States();selection=json.loads((R/'selection.json').read_text());ids=selection['selected_train_indices'];parents=sorted({r['parent_component'] for r in tr.rows})
 shuffled=np.random.default_rng(98090).permutation(parents);folds={p:int(i%5) for i,p in enumerate(shuffled)}
 put('crossfit_folds.json',folds);fitmethods=json.loads((R/'fit_method.json').read_text());methods={m['h']:m for m in fitmethods};receipts=[]
 for h in H:
  ys=[];zs=[];ps=[]
  for i in ids:
   b=tr.get(i,(h,),'short','P1')
   if not b['world_valid'].all():continue
   z,why=st.features(tr,tr.rows[i],h);assert z is not None,why
   ys.append(b['world'].numpy());zs.append(z);ps.append(tr.rows[i]['parent_component'])
  Y=np.stack(ys);Z=np.stack(zs);old=dict(np.load(R/f'R1_h{h}.npz'));alpha=methods[h]['alphas']['R1']['alpha'];out={'full_'+k:old[k] for k in ('w1','ym','zm','zs')};checks=[]
  for fold in range(5):
   mask=np.array([folds[p]!=fold for p in ps]);zm,zs=stats(Z[mask],1e-6);m=fit_r1((Z[mask]-zm)/zs,Y[mask],alpha)
   for k,v in dict(m,zm=zm,zs=zs).items():out[f'f{fold}_{k}']=v.astype('f4')
   fp={p for p,mk in zip(ps,mask) if mk};hp={p for p,mk in zip(ps,mask) if not mk};assert not fp&hp
   checks.append(dict(fold=fold,fit_parents=len(fp),heldout_parents=len(hp),fit_windows=int(mask.sum()),heldout_windows=int((~mask).sum()),parent_overlap=0))
  np.savez_compressed(OUT/f'crossfit_h{h}.npz',**out);receipts.append(dict(h=h,alpha=alpha,windows=len(Y),folds=checks,full_R1_sha=sha(R/f'R1_h{h}.npz')))
  print('crossfit',h,len(Y),flush=True)
 put('CROSSFIT_LOCK.json',dict(brt=now(),fold_seed=98090,folds=5,fit=receipts,normalization='each crossfit model fit-only; full exactly E97-A1 frozen',continuous_rule='10..50 interpolate neighboring fixed-R1 raw predictions using actual q(t+h); 51..100 use frozen global h100 with actual q(t+h); divide original scale(h); no dev fitting',trajectory='all actual source rows offsets0..100, joint/current/history/FK as E97 124D; W(h) reads only offsets<=h; O cannot read any robot token',hashes={p.name:sha(p) for p in OUT.glob('crossfit_h*.npz')}))
 with (T/'notes/e98w_robot_conditioned_20261114.md').open('a') as f:f.write('\n### R1交叉拟合合同锁定\n'+now()+'：5折parent交叉拟合，seed98090；每折仅fit parent拟合均值、尺度及系数，alpha逐h原样沿用E97-A1，不看dev；dev完全读取原训练全量冻结R1。10..50随机时域用相邻固定R1在实际端点q特征上的raw预测插值；51..100用原global R1(+100)作用于实际端点，再除原scale(h)。四个固定h精确等于原R1；不扩alpha/不重新定尺度。条件轨迹读取原行0..100全部实测q/FK，W(h)仅读<=h，O完全隔离。\n')
if __name__=='__main__':main()
