import ast
from pathlib import Path
p=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');text=(p/'code_e101s/run.py').read_text();tree=ast.parse(text);lines=text.splitlines();names=['calibration','calibrate','stagefit','transition','build_aux','emission','boundaries','match','record','aggregate']
header='''from train import *
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from numba import njit
S=[f'S{i:02}' for i in range(1,11)];SI={s:i for i,s in enumerate(S)};SPEC=oldread('spec.json')
stages={pool:oldread('stages_'+pool+'.json') for pool in ['train','dev']}
def truth(s):
 y=np.full(s['n'],-1,int)
 for z in stages[s['pool']]:
  if z['parent_id']==s['parent']:y[z['stage_start']:z['stage_end_exclusive']]=SI.get(z['stage_id'],-1)
 return y
truths={s['parent']:truth(s) for s in sources}
A=np.eye(10,dtype=bool)
for t in SPEC['transitions']:
 if t['source'] in SI and t['target'] in SI:A[SI[t['source']],SI[t['target']]]=1
'''
out=header+'\n'+'\n\n'.join('\n'.join(lines[n.lineno-1:n.end_lineno]) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
(p/'code_e102s/s2.py').write_text(out+'\n')
