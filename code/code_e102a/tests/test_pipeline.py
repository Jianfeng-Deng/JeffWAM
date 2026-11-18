import pytest,json,torch,numpy as np,time
from pathlib import Path
from e86.common import C,R,sha,V1_HASH
from e86.data import Package,verify_file,build_window,CachedDataset,collate
from e86.deploy import wait_for_state

def test_pinned_P3_only_and_counts():
    p=Package(version="V1");assert p.receipt['status']=='ready_frozen_V1';assert p.receipt['source']['sha256']==V1_HASH;assert len(p.entries)==206;assert sum(e['windows'] for e in p.entries.values())==24477;assert all(not e['recovery'] for e in p.entries.values())

def test_P3_candidate_cannot_replace_receipt(tmp_path):
    # The gate requires the receipt hash itself, regardless of a forged `status` field.
    p=Package(version="V1");fake=tmp_path/'P3_ready_V1.json';fake.write_text(json.dumps(dict(p.receipt,status='ready_frozen_V1',source={'path':'candidate','sha256':'bad'})))
    with pytest.raises(ValueError):verify_file(fake,sha(R/'P3_ready_V1.json'))

def test_rev4_all_parent_assignments():
    p=Package(version="V1");assert all(p.split[c['parent_component']]==c['split'] for c in p.captures);tr={c['parent_component'] for c in p.captures if c['split']=='train'};ev={c['parent_component'] for c in p.captures if c['split']!='train'};assert not tr&ev

def test_no_shift_next_feedback_and_history_masks():
    p=Package(version="V1");c=next(c for c in p.captures if c.get('train_ready_p1'));z=p.checked_np(c['numeric_cache']);w=p.checked_np(c['window_cache']);a=int(w['p1_anchor'][0]);r=build_window(c,z,a,w['p1_O_video_rows'][0],w['p1_W_video_rows'][0],'P1');assert np.array_equal(z['original_action_index'][a:a+50],np.arange(a+1,a+51));assert np.array_equal(r['obs_rows'][-1],w['p1_O_video_rows'][0,-1]);assert len(r['history_valid'])==4
    for h,off in enumerate((-24,-18,-12,-6)):
        if a+off<0:assert not r['history_valid'][h]

def test_state_wait_actually_incurred():
    t=time.monotonic();wait=wait_for_state(t+.015);assert wait>.010;assert time.monotonic()-t>=.014

def test_cache_loader_rejects_eval_model_text(tmp_path):
    # Model annotations on eval are explicitly rejected before their use.
    r=dict(canonical_id='c',anchor=10,split='dev',parent_component='devparent')
    ds=CachedDataset([r],[dict(canonical_id='c')],tmp_path,{}, {'c':np.ones((1,768))})
    with pytest.raises(ValueError,match='Model language'):ds[0]

def test_published_windows_preserve_P3_V1():
    path=C/'artifacts/windows_P3_v2.json'
    if not path.exists():pytest.skip('CPU rebuild pending')
    ws=json.loads(path.read_text());p=Package(version="V1");assert len(ws)==24477
    actual={}
    for w in ws:actual.setdefault(w['canonical_id'],set()).add(w['anchor'])
    assert actual=={cid:set(e['window_anchor_raw_rows']) for cid,e in p.entries.items()}


def test_extended_history_wait_cannot_reach_current_future():
    p=Package(version="V1");c=next(c for c in p.captures if c.get('train_ready_p1'));z=p.checked_np(c['numeric_cache']);w=p.checked_np(c['window_cache']);ix=next(i for i,a in enumerate(w['p1_anchor']) if a>=30);a=int(w['p1_anchor'][ix]);z['state_available_time']=z['state_available_time'].copy();z['state_available_time'][a-24]=z['state_available_time'][a]+1
    r=build_window(c,z,a,w['p1_O_video_rows'][ix],w['p1_W_video_rows'][ix],'P1');assert not r['history_valid'][0];assert r['obs_rows'][0]==[-1,-1,-1]
