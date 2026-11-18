import json,numpy as np
from common import *
from scipy.stats import beta
x=json.loads((P/'frozen/pixel_results.json').read_text())
r=next(r for r in x['left_right_switch'] if r['split']=='probe-val' and r['scope']=='gripper' and r['radius_multiplier']==1 and r['disjoint_targets'])
n=r['parents'];assert r['frames']==n and r['both_queries_follow']['estimate']==0
upper=float(beta.ppf(.975,1,n))
atomic(P/'pixel_statistical_limits.json',{'timestamp_brt':brt(),'scope':'trusted both-gripper visible, disjoint r-neighborhood probe-val subset','parents':n,'frames':n,'both_follow_successes':0,'cluster_bootstrap_ci95':r['both_queries_follow']['ci95'],'supplemental_independent_parent_binomial_CP95_upper':upper,'warning':'All-zero empirical cluster bootstrap degenerates to[0,0]; it does not prove population failure or certainty. Binomial supplement assumes independent binary parents, valid here only because one frame per parent. Approximate labels/systematic error still unaccounted. Laterality ability remains inconclusive, no automatic LoRA trigger.'})
append_report('## 像素统计力补充 '+brt()+'\n\n左右trusted且r圆可分子集仅'+str(n)+'帧/'+str(n)+'parent，双句均跟随0次，经验parent bootstrap退化为[0,0]，**不代表总体成功率已精确为0**。仅在此每parent一帧子集，独立parent二项Clopper–Pearson双侧95%上界为'+f'{upper*100:.2f}%'+ '，仍不包含模型GT共有偏差。故左右能力不可判定，不能凭全零bootstrap强行触发LoRA。详见pixel_statistical_limits.json。')
print('PIXEL_UNCERTAINTY',n,'parents; CP95 upper',round(upper,4))
