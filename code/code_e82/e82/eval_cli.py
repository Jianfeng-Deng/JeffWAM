"""Evaluate locked recorded actions or aggregate paired prediction files; no W score selection."""
import argparse,json
from pathlib import Path
import numpy as np
from .evaluation import window_metrics,parent_equal,paired_bootstrap
from .runtime import atomic_json

def evaluate_file(path,scale):
    with np.load(path,allow_pickle=False) as z:
        required={'prediction','truth','window_id','parent_component','session_id','truth_source','split'}
        if not required.issubset(z.files):raise ValueError(f'Missing prediction metadata {required-set(z.files)}')
        if not np.isin(z['truth_source'],['human','recorded_feedback']).all():raise ValueError('Model-generated evaluation truth forbidden')
        if not np.isin(z['split'],['dev','test','exploratory_dev','exploratory_test']).all():raise ValueError('Evaluation split not locked')
        metrics=window_metrics(z['prediction'],z['truth'],scale)
        return {k:parent_equal(v,z['parent_component'].tolist(),z['session_id'].tolist()) for k,v in metrics.items()},z['window_id'].tolist(),z['truth'].copy()

def main():
    p=argparse.ArgumentParser();p.add_argument('--R',required=True);p.add_argument('--P',required=True);p.add_argument('--stats',required=True);p.add_argument('--output',required=True);p.add_argument('--draws',type=int,default=10000);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    stats=json.loads(Path(a.stats).read_text());scale=np.asarray(stats['action_std'])
    r,ri,rt=evaluate_file(a.R,scale);p,pi,pt=evaluate_file(a.P,scale)
    if ri!=pi or not np.array_equal(rt,pt):raise ValueError('Pairing requires identical frozen windows and truth')
    out={}
    for k in r:
        out[k]=dict(R=r[k]['mean'],P=p[k]['mean'],session_parent=paired_bootstrap(r[k],p[k],a.draws),parent=paired_bootstrap(r[k],p[k],a.draws,hierarchical=False))
    atomic_json(a.output,out)
if __name__=='__main__':main()
