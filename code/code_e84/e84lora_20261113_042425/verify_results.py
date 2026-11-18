import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import sys
sys.dont_write_bytecode=True
from common import *
from contract import sha
import numpy as np,torch,collections

def main():
 checks={};sel=json.loads((P/'selection_full_v4.json').read_text());checks['selection_unchanged']=sha(P/'selection_full_v4.json')==sha(OLDP/'selection_full_v4.json');split=json.loads((R/'split/probe_split.json').read_text());items=json.loads((L/'training_items.json').read_text());checks['only_probe_train']=all(split['parents'][r['parent']]=='probe-train' for r in items)
 a=json.loads((L/'lora/training_result.json').read_text());b=json.loads((L/'shuffled/training_result.json').read_text());checks['same768steps']=a['steps']==b['steps']==768;checks['same_config']=a['config']==b['config'];checks['rank8_4layers_qv']=len(a['config']['modules'])==16 and a['config']['rank']==8 and a['config']['trainable_parameters']==196608
 for v in ['lora','shuffled']:
  state=torch.load(L/v/'adapter.pt',map_location='cpu',weights_only=True);checks[v+'_finite_adapter_only']=all(k.endswith(('.A','.B')) and torch.isfinite(t).all().item() for k,t in state.items());checks[v+'_both_towers_changed']=all(any(k.startswith(tower) and k.endswith('.B') and torch.count_nonzero(t)>0 for k,t in state.items()) for tower in ['vision_model','text_model']);checks[v+'_pixel_gt_eval_only']=not json.loads((L/v/'training_manifest.json').read_text()).get('pixel_supervision',False)
  root=R/'features'/RUN/v;receipts=list(root.glob('*_receipt.json'));checks[v+'_542captures']=len(receipts)==542
  for p in receipts:
   r=json.loads(p.read_text());assert sha(r['path'])==r['sha256'];assert r['source_stat_before_after_equal']
   f=np.load(r['path']);assert 'patch_features' not in f.files;assert np.isfinite(f['global_features']).all();f.close()
  f=np.load(root/'pixel_150.npz');checks[v+'_150pixel']=len(f['ids'])==150 and np.isfinite(f['patch_features']).all();f.close()
  assignment=json.loads((L/v/'training_label_assignment.json').read_text());checks[v+'_assignment_train_only']=all(split['parents'][r['parent']]=='probe-train' for r in assignment)
 x=json.loads((L/'lora/training_label_assignment.json').read_text());y=json.loads((L/'shuffled/training_label_assignment.json').read_text());xf=[r['assigned_label'] for r in x if r['kind']=='fk'];yf=[r['assigned_label'] for r in y if r['kind']=='fk'];checks['shuffle_preserves_fk_frequencies']=collections.Counter(xf)==collections.Counter(yf);checks['shuffle_changes_fk_labels']=xf!=yf
 checks['no_resource_stop']=not (P/'resource_stop.json').exists();checks['gpu_lt4h']=sum(json.loads(p.read_text())['elapsed_s'] for p in (P/'gpu_ledger').glob('*.json'))<4*3600
 atomic(P/'verification.json',dict(timestamp_brt=brt(),checks=checks,all_passed=all(checks.values())));assert all(checks.values()),checks;print('VERIFIED',len(checks),'checks')
if __name__=='__main__':main()
