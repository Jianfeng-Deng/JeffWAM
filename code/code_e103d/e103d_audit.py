import os,json,hashlib,collections
from e103d_runtime import *
def main():
 plan=json.loads((CACHE/'plan.json').read_text());audit=collections.defaultdict(collections.Counter);hashes=[]
 for r in plan['records']:
  o=r['original']
  if r['source']=='abc':
   p=Path(o['path']);st=p.stat();assert (st.st_size,st.st_mtime_ns)==(o['size_bytes'],o['mtime_ns']);h=hashlib.sha256();h.update(b'ABC-E89-head-tail-v1\0');h.update(str(st.st_size).encode()+b'\0')
   with p.open('rb') as f:h.update(f.read(65536));h.update(b'\0');f.seek(max(0,st.st_size-65536));h.update(f.read(65536))
   assert h.hexdigest()==o['sha256'],('ABC source drift',p);hashes.append(dict(cid=r['canonical_id'],source='abc',sha256=h.hexdigest(),scope=o['hash_scope']))
  else:
   for cam in ['top','left','right']:
    q=o['small_files_sha256'][cam+'-timestamp.npy'];assert sha(q['path'])==q['sha256']
   hashes.append(dict(cid=r['canonical_id'],source='yam',timestamp_hashes={cam:o['small_files_sha256'][cam+'-timestamp.npy']['sha256'] for cam in ['top','left','right']}))
 for a in plan['audit']:
  s=a['source'];audit[s]['episodes']+=1;audit[s]['candidate_anchors']+=a.get('candidate_anchors',0);audit[s]['valid_anchors']+=a.get('valid',0)
  for k,v in a.get('reasons',{}).items():audit[s][k]+=v
  if a.get('rejected'):audit[s]['episode_rejected']+=1
 refs=[T/'notes'/f for f in ['e89_abc_inventory_20261114.md','e87_yam_inventory_20261113.md','e90_full_v31_p1_20261114.md','e100c_chain_fix_20261115.md']]+sorted((T/'notes').glob('e9[12]*.md'))+[T/'data/e91_abc_lang_v1/lang_abc_v3.jsonl',T/'data/e92_yam_lang_v1/lang_yam_v2.jsonl']
 out=dict(status='passed',source_hash_checks=hashes,timing_summary={s:dict(v) for s,v in audit.items()},refs_sha256={str(p):sha(p) for p in refs},model_text_use='ABC/yam model narratives are not inputs, targets, truth, thresholds or checkpoint selection in this run; fixed generic instruction only; user-mandated six-clip training-pool exclusion applied',clock_limitation='ABC record timestamp has no independent receipt/exposure clock evidence; yam raw timestamps are host receive-time approximation. No manual shift or interpolation.',camera_mapping={'Piper':['cam_high','cam_left_wrist','cam_right_wrist'],'ABC':['/top-camera','/left-wrist-camera','/right-wrist-camera'],'yam':['top','left','right']},camera_evidence='ABC named MCAP topics and E89 calibration/visual audit; yam E87 raw-to-cam_high/left/right conversion schema and wrist views; no geometric equivalence or extrinsics claim',brt=brt())
 atomic(CONTROL/'DATA_AUDIT.json',out);report('## 数据来源与时间闸门审计\n'+json.dumps({k:v for k,v in out.items() if k!='source_hash_checks'},ensure_ascii=False));print(json.dumps(out['timing_summary']),flush=True)
if __name__=='__main__':main()
