from tune import *
def main():
 assert (R/'CONTROLLER_LOCK.json').exists() and not (R/'DEV_ONCE_STARTED.json').exists();lock=read('CONTROLLER_LOCK.json');hl=read('HEADS_LOCK.json');assert sha(R/'final_aux.pkl')==lock['hashes']['aux'] and sha(R/'HEADS_LOCK.json')==lock['hashes']['heads']
 with (R/'final_aux.pkl').open('rb') as f:aux=pickle.load(f)
 models={}
 checks=[]
 for method in ['H0','H1','H2','H3']:
  with (R/(method+'_full.pkl')).open('rb') as f:models[method]=pickle.load(f)
  s=train[0];length=hl['methods'][method]['history_frames'];xx=make(s,np.arange(128),method,length);p,u=predict(models[method],xx);p2,u2=predict(models[method],xx[:64]);err=float(abs(p[:64]-p2).max());assert err<1e-5,(method,err);checks.append(dict(method=method,train_prefix_max_abs=err))
 put('head_causal_contracts.json',checks)
 put('DEV_ONCE_STARTED.json',dict(brt=now(),controller_lock_sha=sha(R/'CONTROLLER_LOCK.json'),model_files={m:sha(R/(m+'_full.pkl')) for m in models},label_basis='model, non-human; user explicitly authorized dev reference',evaluations=1))
 records={m+'__'+i:[] for m in models for i in ['fixed_S01','uniform']};records['E101_S2']=[]
 for s in dev:
  parent=s['parent'];scores={};seqs={};tt=truths[parent]
  for method,a in models.items():
   length=hl['methods'][method]['history_frames'];ps=[];us=[]
   for fr in np.array_split(np.arange(s['n']),max(1,int(np.ceil(s['n']/256)))):
    p,u=predict(a,make(s,fr,method,length));ps.append(p);us.append(u)
   p=np.concatenate(ps);u=np.concatenate(us);scores[method+'_prob']=p;scores[method+'_unknown']=u;e=emission(p,aux[method])
   for initial in ['fixed_S01','uniform']:
    name=method+'__'+initial;seq=hmm(e,aux[method],**lock['selected'][name]['config']);seqs[name]=seq;records[name].append(dict(parent=parent,**record(seq,tt)))
  old=np.load(T/'results/e101s_20261115_01'/(parent+'_frozen_sequences.npz'));assert np.array_equal(old['truth'],tt);seqs['E101_S2']=old['S2'];records['E101_S2'].append(dict(parent=parent,**record(old['S2'],tt)))
  np.savez_compressed(R/(parent+'_dev_frozen_scores.npz'),**scores);np.savez_compressed(R/(parent+'_dev_frozen_sequences.npz'),truth=tt,**seqs);status('dev_frozen',parent=parent)
 put('dev_frozen_parent_statistics.json',records);put('DEV_FROZEN_COMPLETE.json',dict(brt=now(),parents=20,frames=sum(s['n'] for s in dev),evaluations=1));report('## dev冻结完成 '+now()+'\n20 parent、55754帧；四种头各输出一次冻结分数，8个在线S2变体一次回放。E101-S逐parent冻结预测原位复用作为配对参考，没有重训旧模型。所有后续统计仅从冻结产物读取，不再选择配置。');status('DEV_FROZEN_COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception:put('DEV_FAILURE_'+str(int(time.time()))+'.json',dict(brt=now(),traceback=traceback.format_exc()));raise
