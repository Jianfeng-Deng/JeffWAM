from common import *
import numpy as np,collections,pickle
SPEC=read('spec.json');PIDS=[p['id'] for p in SPEC['predicates']];PI={p:i for i,p in enumerate(PIDS)}
def replay(prob,unknown,thresholds,persist):
 state='S01';out=[];events=set();last=[None]*21;count=np.zeros(21,int)
 def guard(g,ans):
  if 'all' in g:return all(guard(v,ans) for v in g['all'])
  if 'any' in g:return any(guard(v,ans) for v in g['any'])
  if 'predicate' in g:return ans[PI[g['predicate']]]==g['equals']
  if 'event' in g:return g['event'] in events
  if 'context' in g:return g['context']=='batch_continue_enabled'
  raise ValueError(g)
 for ps,us in zip(prob,unknown):
  current=['unknown' if u>=.5 else 'yes' if p>=t else 'no' for p,u,t in zip(ps,us,thresholds)];answers=[]
  for j,a in enumerate(current):
   count[j]=count[j]+1 if a==last[j] else 1;last[j]=a;answers.append(a if count[j]>=persist else 'unknown')
  # prior-frame event only; don't use current P071 in the same transition guard.
  cleared=False
  candidates=[t for t in SPEC['transitions'] if t['source']==state and guard(t['guard'],answers)]
  if candidates:
   priority=min(t['priority'] for t in candidates);top=[t for t in candidates if t['priority']==priority]
   if len({t['target'] for t in top})==1:
    t=top[0];state=t['target']
    if t['kind'] in ('failure','next_cycle','retry','abandon') or 'increment_attempt_id' in t['effects']:events.clear();cleared=True
  if not cleared and answers[PI['P101']]!='yes' and answers[PI['P071']]=='yes':events.add('P071_confirmed_earlier_this_attempt')
  out.append(state)
 return out

def compress(seq):return [s for i,s in enumerate(seq) if i==0 or s!=seq[i-1]]
def levenshtein(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  nxt=[i]
  for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
  row=nxt
 return row[-1]
def truth(pool,parent,n):
 seq=['unknown']*n
 for s in read('stages_'+pool+'.json'):
  if s['parent_id']==parent:seq[s['stage_start']:s['stage_end_exclusive']]=[s['stage_id']]*(s['stage_end_exclusive']-s['stage_start'])
 return seq

def boundary_match(pred,true):
 pp=[(i,pred[i-1],pred[i]) for i in range(1,len(pred)) if pred[i]!=pred[i-1]];tt=[(i,true[i-1],true[i]) for i in range(1,len(true)) if true[i]!=true[i-1] and 'unknown' not in (true[i-1],true[i])]
 # Order-preserving exact transition-type matching; maximize matches then minimize total error.
 n,m=len(pp),len(tt);dp=[[None]*(m+1) for _ in range(n+1)];dp[0][0]=(0,0,[])
 for i in range(n+1):
  for j in range(m+1):
   if i==j==0:continue
   options=[]
   if i:options.append(dp[i-1][j])
   if j:options.append(dp[i][j-1])
   if i and j and pp[i-1][1:]==tt[j-1][1:]:
    v=dp[i-1][j-1];options.append((v[0]+1,v[1]-abs(pp[i-1][0]-tt[j-1][0]),v[2]+[(i-1,j-1)]))
   dp[i][j]=max(options,key=lambda x:(x[0],x[1]))
 matches=dp[n][m][2];signed=[(pp[i][0]-tt[j][0])/30 for i,j in matches]
 return dict(true_boundaries=m,pred_boundaries=n,matched=len(matches),missed=m-len(matches),extra=n-len(matches),signed_error_seconds=signed)
def eval_seq(seq,true):
 a=compress(seq);b=compress([s for s in true if s!='unknown']);ed=levenshtein(a,b);return dict(edit_distance=ed,normalized_edit_distance=ed/max(len(b),1),pred_sequence=a,true_sequence=b,**boundary_match(seq,true))
def lock():
 audit=read('internal_validation.json');th=audit['d']['thresholds'];scores=[]
 for persist in (1,3,6):
  vals=[]
  for s in read('sources.json'):
   if not s['validation']:continue
   z=np.load(R/(s['parent']+'_val_scores.npz'));seq=replay(z['prob'],z['unknown'],th,persist);rr=eval_seq(seq,truth('train',s['parent'],s['n']));vals.append(dict(parent=s['parent'],**rr))
  scores.append(dict(persistence=persist,mean_normalized_edit_distance=float(np.mean([r['normalized_edit_distance'] for r in vals])),parents=vals))
 best=min(scores,key=lambda s:s['mean_normalized_edit_distance']);put('B4_LOCK.json',dict(brt=now(),method='d',thresholds=th,persistence=best['persistence'],initial_stage='S01',batch_continue=True,candidates=scores,selection='mean normalized stage edit distance internal val only'))
 report('## B4 训练内锁定 '+now()+'\n固定 d，persist='+str(best['persistence'])+'；候选内部验证编辑距离：'+str([(v['persistence'],v['mean_normalized_edit_distance']) for v in scores])+'。初态 S01，不读取 dev 初态；不使用真值 cycle/attempt 事件。')
def evaluate():
 lock=read('B4_LOCK.json');records=[]
 for s in read('sources.json'):
  if s['pool']!='dev':continue
  z=np.load(R/(s['parent']+'_scores.npz'));seq=replay(z['d_prob'],z['d_unknown'],lock['thresholds'],lock['persistence']);r=eval_seq(seq,truth('dev',s['parent'],s['n']));records.append(dict(parent=s['parent'],**r))
  with (R/(s['parent']+'_b4.json')).open('x') as f:json.dump(dict(parent=s['parent'],stage_sequence=seq),f)
 signed=[e for r in records for e in r['signed_error_seconds']];ab=np.abs(signed);total=sum(r['true_boundaries'] for r in records);matched=sum(r['matched'] for r in records)
 summary=dict(label_basis='基于模型标注的 dev 真值',parents=len(records),mean_edit_distance=float(np.mean([r['edit_distance'] for r in records])),mean_normalized_edit_distance=float(np.mean([r['normalized_edit_distance'] for r in records])),matched=matched,true_boundaries=total,pred_boundaries=sum(r['pred_boundaries'] for r in records),boundary_abs_error_seconds_median=float(np.median(ab)) if len(ab) else None,boundary_abs_error_seconds_p90=float(np.quantile(ab,.9)) if len(ab) else None,early_ratio_among_matched=float(np.mean(np.array(signed)<0)) if len(signed) else None,missed_ratio=sum(r['missed'] for r in records)/max(total,1),extra_predicted_boundaries=sum(r['extra'] for r in records),matching='order-preserving exact transition type max-cardinality min-total-absolute-time; no error cutoff; point boundaries; unmatched retained',limitations='design S06->S07 P062 guard differs from adjudicated visible regrasp start; controller lacks independent target identity/context sensor; training labels do not cover all next-stage predicates')
 # Parent bootstrap for controller summaries; matched-only temporal errors always show denominator.
 rng=np.random.default_rng(9903);boot=[]
 for _ in range(2000):
  rr=[records[i] for i in rng.integers(len(records),size=len(records))];ee=np.abs([e for r in rr for e in r['signed_error_seconds']]);ss=[e for r in rr for e in r['signed_error_seconds']];den=sum(r['true_boundaries'] for r in rr)
  boot.append([np.mean([r['edit_distance'] for r in rr]),np.mean([r['normalized_edit_distance'] for r in rr]),np.median(ee) if len(ee) else np.nan,np.quantile(ee,.9) if len(ee) else np.nan,np.mean(np.array(ss)<0) if len(ss) else np.nan,sum(r['missed'] for r in rr)/max(den,1)])
 for j,k in enumerate(('mean_edit_distance','mean_normalized_edit_distance','boundary_abs_error_seconds_median','boundary_abs_error_seconds_p90','early_ratio_among_matched','missed_ratio')):summary[k+'_ci95']=np.nanquantile(np.array(boot)[:,j],[.025,.975]).tolist()
 put('B4_results.json',dict(summary=summary,parents=records));report('## B4 dev 逐源帧回放 '+now()+'\n'+json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':lock() if sys.argv[1]=='lock' else evaluate()
