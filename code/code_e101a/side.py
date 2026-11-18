import sys,os
sys.dont_write_bytecode=True
sys.path.insert(0,'/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e101a')
from diag import *
logs=[json.loads(l) for l in (T/'results/e100c_c2_s90_20261115_03/updates.jsonl').read_text().splitlines()];out={};bins=[]
for mode in ['all','action_only','sync','clean','independent']:
 rr=[r for r in logs if mode=='all' or r['mode']==mode];s=np.array([r['step'] for r in rr]);a=np.array([r['action'] for r in rr]);late=s>3000
 out[mode]=dict(n=len(rr),first100=float(a[:100].mean()),last100=float(a[-100:].mean()),slope_per1000=float(np.polyfit(s,a,1)[0]*1000),last2000_slope_per1000=float(np.polyfit(s[late],a[late],1)[0]*1000),action_fraction_loss=float(np.mean([r['action']/r['loss'] for r in rr])))
 for st in range(0,5000,100):
  aa=a[(s>st)&(s<=st+100)]
  if len(aa):bins.append(dict(mode=mode,step=st+50,action=float(aa.mean())))
save('training_slopes.json',out);save('training_bins.json',bins)
report('## 4 训练日志审计\nC2总损失所有模式均为 `la + 0.1*(world + 0.1*cos + 0.1*adapter + cw*chain) + 0.01*align`。action_only 的 world/cos/adapter/chain 为0；其余三个模式 world直接系数0.1，cos/adapter各0.01，chain系数0.1*cw（上限0.01）。clean另做t=0.5动作FM前向，因此并未放弃动作损失；四模式la直接系数均1，不是只有25%的更新训练动作。动作时间权重前10×4、中15×2、后25×1；夹爪2/14维，无单独加权。\n\n|模式|更新|首100/末100动作loss|全程斜率/1000步|后2000斜率/1000步|action/total均值|\n|---|---:|---|---:|---:|---:|\n'+'\n'.join(f"|{k}|{v['n']}|{v['first100']:.5f}/{v['last100']:.5f}|{v['slope_per1000']:.5f}|{v['last2000_slope_per1000']:.5f}|{v['action_fraction_loss']:.3f}|" for k,v in out.items())+'\n斜率是各模式训练噪声样本FM损失的描述性OLS，不是物理动作误差；固定5000终点未选型。分100步曲线见training_bins.json。')
# Capture provenance before any optional supplement; read-only source hashes.
paths=[T/'results/e100c_c2_s90_20261115_03/ckpt_05000.pt',T/'code_e100c/e99w2_data.py',T/'code_e100c/e99w2_model.py',T/'code_e100c/e99w2_objective.py',T/'code_e100c/e98w_core.py']
save('provenance.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
status('complete')
