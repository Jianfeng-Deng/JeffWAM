import json,collections,numpy as np
from pathlib import Path
from .common import *
from .data import *
def main():
    dest=C/'artifacts';dest.mkdir(exist_ok=True);planpath=dest/'cache_plan_v1.json';assert not planpath.exists()
    p=Package();p1=p.windows('P1');p3=p.windows('P3');stats=fit_stats(p,p3);atomic(dest/'stats_v1.json',stats)
    atomic(dest/'windows_P1.json',p1);atomic(dest/'windows_P3.json',p3)
    allw=p1+p3;needed=defaultdict(lambda:[set(),set(),set()])
    for w in allw:
        for grid in (w['obs_rows'],w['future_rows']):
            for ids in grid:
                for k,r in enumerate(ids):
                    if r>=0:needed[w['canonical_id']][k].add(r)
    selected=[];frames=0
    for c in p.captures:
        cid=c['canonical_id']
        if cid not in needed:continue
        cams=[]
        for k,cam in enumerate(CAMS):
            route=next(x for x in p.routes[cid]['cameras'] if x['camera']==cam);ids=sorted(needed[cid][k]);frames+=len(ids)
            cams.append(dict(camera=cam,path=route['path'],rows=ids))
        selected.append(dict(canonical_id=cid,parent_component=c['parent_component'],split=c['split'],cameras=cams))
    estimate=frames*8*768*2+frames*8+300_000_000
    assert estimate<=150_000_000_000,'Cache estimate >150GB'
    plan=dict(schema='e86-cache-plan-v1',created_brt=brt(),refs=p.refs,captures=selected,unique_camera_frames=frames,estimated_bytes=estimate,P1_windows=len(p1),P3_windows=len(p3),stats_sha256=sha(dest/'stats_v1.json'))
    atomic(planpath,plan)
    atomic(dest/'data_audit.json',dict(P1_windows=len(p1),P3_windows=len(p3),P1_parents=len(set(w['parent_component'] for w in p1)),P3_parents=len(set(w['parent_component'] for w in p3)),demo3_P3_windows=sum(w['demo3'] for w in p3),unique_camera_frames=frames,estimated_bytes=estimate,refs=p.refs,all_dependencies_verified=True))
    append_report('### 数据与缓存预算（'+brt()+'）\n'+json.dumps(json.loads((dest/'data_audit.json').read_text()),ensure_ascii=False)+'\n缓存=每相机每源帧固定2×4×768 fp16；剔除padding权重。O当前每相机learned8、四历史每帧三相机合并learned8；W=三未来×三相机×8固定目标。E84大部分仅global，不能冒充8空间token，故不直接复用。')
    print(json.dumps({k:v for k,v in plan.items() if k!='captures'},ensure_ascii=False))
if __name__=='__main__':main()
