import os,sys,inspect,json,traceback
from pathlib import Path
from e103d_runtime import *
from e102x_setup import datasets
import e102a_eval as ev
from e103d_model import WorldModel

def evaluate(group):
 lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());CKPT=Path(lock['checkpoints'][group]);ART=CONTROL/('eval_'+group);ART.mkdir(exist_ok=False)
 src=inspect.getsource(ev.evaluate).replace("assert len(f['train64'])==64 and len(f['dev'])==48","assert len(f['dev'])==len(lock['entries'])")
 start=src.index(' c0=json.loads(');end=src.index(' cp=torch.load',start);src=src[:start]+src[end:]
 src=src.replace("RUN/'ckpt_02000.pt'",'CKPT').replace("assert cp['step']==STEPS;",'')
 src=src.replace("[('dev',dev,f['dev']),('train64',tr,f['train64'])]","[('dev',dev,f['dev'])]")
 src=src.replace('     actions.append(item)',"     item.update(cid=r['canonical_id'],anchor=r['anchor'],true_event_ids=[[r['canonical_id'],k,r['anchor']+t,d] for k in range(2) for t,d in events(labels[:,k],init[k])]);actions.append(item)")
 src=src.replace('for h in (25,50,100):',"for h in (10,25,50,100):\n       if h not in entry['horizons']:continue")
 src=src.replace('seq=qq[a+1:a+101];assert len(seq)==100','seq=qq[np.minimum(a+np.arange(1,101),len(qq)-1)]')
 src=src.replace('       sources=dict(gt_action','       feat[~valid.numpy()]=0\n       sources=dict(gt_action')
 start=src.index("   b=move(dev.batch([f['dev'][0]");end=src.index(' finally:guard.finish()',start);src=src[:start]+src[end:]
 src=src.replace("f=json.loads((CONTROL/'fixed_windows.json').read_text())","f=dict(dev=lock['entries'])")
 ns=dict(ev.__dict__);ns.update(CONTROL=CONTROL,ART=ART,Guard=lambda *a:Guard('eval_'+group,600,6144),lock=lock,load=datasets,CKPT=CKPT,GROUP=group,WorldModel=WorldModel)
 (ART/'evaluator.py').write_text(src);exec(compile(src,str(ART/'evaluator.py'),'exec'),ns);ns['evaluate']()
if __name__=='__main__':
 try:evaluate(sys.argv[1])
 except BaseException:atomic(CONTROL/('eval_'+sys.argv[1]+'_error.json'),dict(traceback=traceback.format_exc(),brt=brt()));raise
