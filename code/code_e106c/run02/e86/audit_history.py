"""Mask-only overlay: preserve all anchors, actions, labels and parent assignments."""
import json,numpy as np
from collections import defaultdict
from .common import *
from .data import Package

def audit(ws,captures):
    by=defaultdict(list);cs={c['canonical_id']:c for c in captures}
    for r in ws:by[r['canonical_id']].append(r)
    changed=0
    for cid,rs in by.items():
        z=dict(np.load(cs[cid]['numeric_cache'],allow_pickle=False));av=z['state_available_time'];seg=z['continuous_segment']
        for r in rs:
            a=r['anchor']
            for k,off in enumerate((-24,-18,-12,-6)):
                if not r['history_valid'][k]:continue
                j=a+off;valid=j>=0 and seg[max(j,0)]==seg[a] and np.isfinite(av[max(j,0)]) and av[max(j,0)]<=av[a]+1e-7
                if valid and 'state_source_brackets' in z:
                    br=z['state_source_brackets'][j];valid=bool(np.isfinite(br).all() and np.max(br[:,1]-br[:,0])<=.075)
                if not valid:r['history_valid'][k]=False;r['obs_rows'][k]=[-1,-1,-1];changed+=1
    return ws,changed

def main():
    p=Package();records=[]
    for stage in ('P1','P3'):
        old=C/f'artifacts/windows_{stage}_v2.json';new=C/f'artifacts/windows_{stage}_v3.json';assert not new.exists();ws,n=audit(json.loads(old.read_text()),p.captures);atomic(new,ws);records.append(dict(stage=stage,windows=len(ws),additional_history_masks=n,path=str(new),sha256=sha(new),source_sha256=sha(old)))
    d=C/'artifacts/dev_v1';old=d/'windows.json';new=d/'windows_v2.json';assert not new.exists();ws,n=audit(json.loads(old.read_text()),json.loads((d/'captures.json').read_text()));atomic(new,ws);records.append(dict(stage='dev',windows=len(ws),additional_history_masks=n,path=str(new),sha256=sha(new),source_sha256=sha(old)))
    atomic(C/'artifacts/READY_windows_v3.json',dict(status='ready',files=records,refs=p.refs,rule='history available time <= current decision; finite bounded brackets; mask only, no anchor/action/label shift',updated_brt=brt()));print(json.dumps(records),flush=True)
if __name__=='__main__':main()
