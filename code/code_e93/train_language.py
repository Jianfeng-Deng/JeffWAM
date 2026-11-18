from common import *
import traceback
started=time.monotonic()
try:
 m=model_cpu();train,tr,dev=get_data();tr.base.text_cache=train.base.text_cache;by=collections.defaultdict(list)
 for i,r in enumerate(tr.rows):by[r['parent_component']].append(i)
 ids=[vs[len(vs)//2] for p,vs in sorted(by.items())];fixed={(r['index'],r['h']):r for r in json.loads((R/'train64_v1_records.json').read_text())};rs=[];fs=[];covered=0
 with torch.inference_mode():
  for j,i in enumerate(ids):
   row=tr.rows[i];covered+=(row['canonical_id'],row['anchor']) in train.base.text_cache
   for h in H:
    if (i,h) not in fixed:continue
    b=tr.batch([i],(h,),'instruction','P1');pred=m.rollout(b,4,9001+i);rs.append(rec(row,h,pred,b,index=i));fs.append(fixed[i,h])
   atomic(R/'train_language_status.json',dict(status='evaluating',done=j+1,total=len(ids)))
 atomic(R/'train_language_results.json',dict(original_instruction=summary(rs),fixed_human_short=summary(fs),parents=len(ids),model_instruction_windows=covered,records=rs,rule='Same64 training parents, one existing midpoint selected row/parent, paired original seeds; original per-anchor training instruction only in TRAIN. No language added to dev/test.'))
 atomic(R/'train_language_status.json',dict(status='completed',seconds=time.monotonic()-started,GPUh=0));print('completed',len(ids),covered)
except BaseException as e:atomic(R/'train_language_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
