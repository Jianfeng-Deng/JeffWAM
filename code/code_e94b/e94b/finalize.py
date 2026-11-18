from .common import *
import numpy as np
from .metrics import aggregate

def table(label,s):
 out=['### '+label,'','|+帧|窗/parent|模型MSE|复制MSE|线性MSE|skill % [95% CI]|残差cos|','|---|---|---|---|---|---|---|']
 for h in H:
  d=s[str(h)];v=d['values'];ci=d['ci95']['skill_copy'];out.append(f'|{h}|{d["windows"]}/{d["parents"]}|{v["model"]:.6f}|{v["copy"]:.6f}|'+(f'{v["linear"]:.6f}' if 'linear' in v else '见逐窗/本目标基线')+f'|{v["skill_copy"]*100:.3f} [{ci[0]*100:.3f}, {ci[1]*100:.3f}]|{v["cosine"]:.5f}|')
 return '\n'.join(out)
def paired_ridge():
 sig=json.loads((E93/'linear_records.json').read_text());dino=json.loads((R/'ridge_8_records.json').read_text());out={}
 for h in H:
  ss={};dd={}
  for records,by in [(sig,ss),(dino,dd)]:
   rows=[r for r in records if r['h']==h]
   for p in sorted({r['parent'] for r in rows}):by[p]=[np.mean([r[k] for r in rows if r['parent']==p]) for k in ('model','copy','cosine')]
  assert set(ss)==set(dd);parents=sorted(ss);a=np.array([ss[p] for p in parents]);b=np.array([dd[p] for p in parents]);rng=np.random.default_rng(90);ix=rng.integers(len(a),size=(2000,len(a)));ab=a[ix].mean(1);bb=b[ix].mean(1);delta=(1-b[:,0].mean()/b[:,1].mean())-(1-a[:,0].mean()/a[:,1].mean());ds=(1-bb[:,0]/bb[:,1])-(1-ab[:,0]/ab[:,1]);dc=bb[:,2]-ab[:,2];out[str(h)]=dict(skill_delta_percentage_points=float(delta*100),skill_delta_CI95_percentage_points=(np.quantile(ds,[.025,.975])*100).tolist(),cos_delta=float(b[:,2].mean()-a[:,2].mean()),cos_delta_CI95=np.quantile(dc,[.025,.975]).tolist(),paired_parents=len(parents))
 atomic(R/'DINO_vs_Sig_ridge_paired_comparison.json',out);return out

def finalize():
 assert json.loads((R/'pipeline_status.json').read_text())['stage']=='completed_training_evaluation'
 g=json.loads((R/'gpu_ledger.json').read_text());assert not g.get('active') and g['elapsed_s']<=14400
 protected=json.loads((R/'protected_inputs.json').read_text());changed=[p for p,h in protected.items() if sha(p)!=h];assert not changed
 # Original snapshots from before any E94b edits are kept separately; all old code and E93 windows stay read-only.
 oldck=T/'code_e90/artifacts/e90_p1_20261114_01/ckpt_05000.pt';assert sha(oldck)=='45164cfb86483709892d5c59a55fb830484cb5923eb7c018579595c826abbc49'
 actual=sum(p.stat().st_size for root in (C,CACHE,R) for p in root.rglob('*') if p.is_file());cumulative=actual+1046032857+268435456;assert cumulative<=30e9
 receipt=dict(status='COMPLETE',finished_brt=brt(),GPU_seconds=g['elapsed_s'],GPUh=g['elapsed_s']/3600,new_retained_bytes=actual,cumulative_written_upper_account_bytes=cumulative,deleted_setup_copy_bytes=1046032857,control_file_rewrites_report_log_and_local_small_files_allowance=268435456,failed_extract_partial_preserved=True,protected_files_checked=len(protected),changed_protected_files=changed,old_checkpoint_sha256=sha(oldck),license='Meta DINOv3 custom; internal research only',limits=dict(GPUh=4,cumulative_bytes=30_000_000_000))
 atomic(R/'final_resource_receipt.json',receipt)
 report('## 6 最终统一评估表\n各表MSE只在本目标归一化空间解释；跨SigLIP2/DINOv3仅比较skill、方向cos。95%区间按parent等权配对bootstrap2000 seed90；未按dev或模型标注选checkpoint。')
 for branch in ('b1','b2'):
  dest=R/('e94b_'+branch+'_20261114_01')
  for pool in ('dev','train64'):
   summaries=json.loads((dest/'evaluation'/(pool+'_summary.json')).read_text())
   for name,s in summaries.items():report(table(branch+' / '+pool+' / '+name,s))
  gates={name:json.loads((dest/('adapter_'+name+'.json')).read_text()) for name in ('initial','chain_boundary')};summ=json.loads((dest/'train_summary.json').read_text());report('## 4 '+branch+' 适配器与链式结果\n'+json.dumps(dict(gates=gates,chain_updates=summ['chain_updates'],adapter_gate_disabled=summ['adapter_gate_disabled']),ensure_ascii=False))
 for grid in (8,49):report(table('DINO岭回归 '+str(grid)+'token / dev',json.loads((R/f'ridge_{grid}_summary.json').read_text())))
 diff=paired_ridge();report('## 6 DINO/Sig 岭回归可预测性比较\n'+json.dumps(diff,ensure_ascii=False,indent=2)+'\n差值以相同parent配对抽样，不比较跨空间MSE。7×7诊断输入为SigLIP2固定2×4双线性插值，对应位置诊断与2×4原协议有此明确区别；+100始终相同2×4全局目标。')
 report('## 7 最终资源与边界收据\n'+json.dumps(receipt,ensure_ascii=False,indent=2))
 atomic(R/'E94B_FINAL_STATUS.json',receipt)
 print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':finalize()
