from e97a_core import *
import h5py
caps=json.loads((E93/'dev/captures_full.json').read_text());dest=R/'dev_states';dest.mkdir(exist_ok=False);receipts=[]
for c in caps:
 candidates=[Path(p.replace('/mnt/workspace/colleague-a/Datasets/piper','/mnt/workspace/dataset/piper',1)) for p in [c['source_path']]+c.get('duplicate_aliases',[])]
 p=next((p for p in candidates if p.is_file()),None)
 if p is None:receipts.append(dict(canonical_id=c['canonical_id'],status='unavailable',candidates=list(map(str,candidates))));continue
 before=p.stat();z=dict(np.load(c['numeric_cache'],allow_pickle=False))
 with h5py.File(p,'r') as f:
  assert not f.attrs.get('sim',False);q=f['observations/qpos'][:];raw=f['observations/qpos_raw'][:];t=f['timestamp'][:];ct=f['camera_timestamps'][:];recv=f['camera_received_timestamps'][:];br=f['state_source_brackets'][:]
 digest=hashlib.sha256(t.tobytes()+raw.tobytes()).hexdigest()
 assert digest==c.get('raw_time_state_sha256',c['canonical_id'].removeprefix('piper_')) or digest.startswith(c['canonical_id'].removeprefix('piper_')),(c['canonical_id'],'state identity mismatch')
 checks={}
 for name,actual in [('timestamp',t),('camera_exposure_timestamp',ct),('camera_received_timestamp',recv),('state_available_time',np.maximum(t,np.max(br,axis=(1,2))))]:
  checks[name]=np.array_equal(z[name],actual);assert checks[name],(c['canonical_id'],name,'not exact')
 for name,actual in [('q',q),('qpos_raw',raw)]:
  if name in z:checks[name]=np.array_equal(z[name],actual);assert checks[name],(c['canonical_id'],name)
 assert q.shape==raw.shape==(len(t),14) and br.shape==(len(t),14,2)
 out=dest/(c['canonical_id']+'.npz');np.savez_compressed(out,q=q,qpos_raw=raw,state_source_brackets=br)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 receipts.append(dict(canonical_id=c['canonical_id'],parent=c['parent_component'],status='restored_exact_identity_and_timing',source=str(p),source_stat=[before.st_size,before.st_mtime_ns],raw_time_state_sha256=digest,checks=checks,rows=len(t),supplement=str(out),sha256=sha(out)))
save('dev_state_restore_receipt.json',receipts)
assert len(receipts)==27 and all(r['status']=='restored_exact_identity_and_timing' for r in receipts)
# First attempt ended in coverage/preflight, before any predictions, scores, or CUDA initialization.
assert not (R/'gpu_ledger.json').exists();assert not list(R.glob('dev_forward_snapshot*'))
for fn in ('DEV_ONCE_LOCK.json','evaluation_failure.json','state_coverage_dev.json','waiter_status.json'):
 p=R/fn
 if p.exists():p.rename(R/('preflight_01_'+fn))
with (T/'notes/e97a_robot_env_decomp_20261114.md').open('a') as f:f.write('\n### dev 状态预检恢复（未产生预测/评分）\n首次预检发现 E93 图像专用 numeric 缓存省略 q/bracket，因无可配对数组在 stack 前停止，0GPUh、0模型前向、0指标。原 manifest 的 colleague-a/Datasets/piper 路径已迁至 /mnt/workspace/dataset/piper；只读原录制恢复了27/27 capture 的 q、signed raw grip、state bracket。所有 timestamp、camera exposure/receipt、state_available_time 与冻结缓存逐元素完全相等；t+rawq 的 SHA 与 canonical identity 完全匹配，已有 q/rawq 也逐元素相等。只补小型独立数值副本，不改窗口/原缓存/语言/系数/超参。原预检锁和失败回执保留 preflight_01_*；随后才执行唯一一次预测评分。详见 dev_state_restore_receipt.json。\n')
print('restored',len(receipts),'bytes',sum(p.stat().st_size for p in dest.glob('*.npz')))
