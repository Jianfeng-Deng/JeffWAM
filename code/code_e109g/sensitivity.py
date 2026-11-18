import os,sys,json,collections
from pathlib import Path
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');sys.dont_write_bytecode=True
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');O=T/'results/e109g_generalization_20261116_01';R=T/'notes/e109g_gripper_generalization_20261116.md';sys.path.insert(0,str(T/'code_e108g'));import analyze as ev
import numpy as np

def main():
 assert (O/'COLLECT_COMPLETE.json').exists();m=json.loads((O/'METRICS.json').read_text());truth=json.loads((O/'WINDOW_TRUTH.json').read_text());audit=json.loads((O/'CACHE_AUDIT.json').read_text());affected={x['cid'] for x in audit['missing']};badpars={r['parent'] for r in truth['train_aligned'] if r['cid'] in affected};allpars=sorted({r['parent'] for r in truth['train_aligned']});goodpars=[p for p in allpars if p not in badpars];idx=np.random.default_rng(90).integers(len(goodpars),size=(2000,len(goodpars)));result={};lines=['## CPU补帧数值与只用旧缓存的敏感性复核','补提158/16692个观测帧（0.9466%），使用本地既有SigLIP2权重/原预处理、CPU float32后fp16 roundtrip；34个已缓存参照帧CPU对旧GPU缓存最大绝对差0.046875，最大逐帧平均绝对差0.004445。不同精度路径并非逐元素相同，不能声称aligned完全复现GPU编码数值；默认模型原211窗基线自身已逐帧/逐事件复现。','不改主评估窗口，另将15个涉及补帧的训练parent整体排除，在剩余130parent上做同parent协议差敏感性复核；这是覆盖限制附表，不是重新选择评估集或checkpoint。','|checkpoint|纯旧缓存train16 F1@10|纯旧缓存aligned F1@10|二者配对差[95%CI]|完整dev F1@10|','|---:|---:|---:|---|---:|']
 for st in ['2000','5000','10000','15000','20000']:
  rec={}
  for s in ['train16','train_aligned']:
   raw=m[st][s]['10']['parent_raw'];rec[s]=np.array([2*raw[p]['tp']/(raw[p]['pred']+raw[p]['true']) if raw[p]['pred']+raw[p]['true'] else 0 for p in goodpars])
  d=rec['train16']-rec['train_aligned'];ci=np.quantile(d[idx].mean(1),[.025,.975]);result[st]=dict(train16=float(rec['train16'].mean()),aligned=float(rec['train_aligned'].mean()),protocol_delta=float(d.mean()),paired_ci95=ci.tolist(),dev=m[st]['dev']['10']['f1']);v=result[st];lines.append(f"|{st}|{v['train16']:.6f}|{v['aligned']:.6f}|{d.mean():+.6f} {ci.tolist()}|{v['dev']:.6f}|")
 # Independent re-score all cached output logits using untouched E108 event extraction/matcher.
 for st in ['2000','5000','10000','15000','20000']:
  for s in ['train16','train_aligned','dev']:
   logits=np.load(O/f'logits_{st}_{s}.npz')['logits'];ss=(logits>0).astype(int);check=ev.score(truth[s],list(ss))
   for t in ['5','10','20']:
    for k in ['f1','false_per_min','recall','matched','true','pred']:assert check[t][k]==m[st][s][t][k],(st,s,t,k)
 lines+=['全部五点×三个窗口集合×三容差已从保存logits独立重算，F1/召回/假检与TP/真/预测数逐项一致。旧缓存130parent附表若与全145parent口径方向一致，说明主口径结论不由CPU补帧少数parent单独驱动；仍不能给出全GPU补帧的精确反事实数值。']
 (O/'CACHED_ONLY_SENSITIVITY.json').write_text(json.dumps(dict(excluded_parents=sorted(badpars),retained_parents=goodpars,metrics=result,logit_rescore_exact=True),indent=2))
 with R.open('a') as f:f.write('\n'+'\n'.join(lines)+'\n')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
