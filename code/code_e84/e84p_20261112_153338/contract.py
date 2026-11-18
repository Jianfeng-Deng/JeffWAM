import hashlib, json, pathlib
from common import *
E=T/'data/e82_piper_v1'
def sha(p): return hashlib.file_digest(pathlib.Path(p).open('rb'),'sha256').hexdigest()
def contracts():
    ready=json.loads((R/'READY_split.json').read_text()); assert ready['status']=='locked'
    for item in ready['files']: assert sha(item['path'])==item['sha256'],item['path']
    split=json.loads(pathlib.Path(ready['files'][0]['path']).read_text())
    assert split['source_interface_revision']==3 and split['model_results_seen'] is False
    assert sha(split['source_split_path'])==split['source_split_sha256']
    assert sha(E/'READY_parent_split.json')==split['source_receipt_sha256']
    original=json.loads(pathlib.Path(split['source_split_path']).read_text())['components']
    assert all(original.get(p)=='train' for p in split['parents'])
    assert set(split['parents'].values())=={'probe-train','probe-val'}
    data=json.loads((E/'READY_data.json').read_text()); assert data['parent_split']['interface_revision']==3
    assert sha(E/'READY_parent_split.json')==data['parent_split']['sha256']
    wanted={E/'manifest/captures_final.jsonl',E/'manifest/video_routes.jsonl'}
    for entry in data['files']:
        if pathlib.Path(entry['path']) in wanted:
            assert sha(entry['path'])==entry['sha256'];wanted.remove(pathlib.Path(entry['path']))
    assert not wanted,'E82 data receipt lacks current capture/route hash'
    allowed={c['canonical_id']:c for c in split['captures']}
    def rows(name):
        out={}
        for line in (E/'manifest'/name).open():
            item=json.loads(line)
            if item['canonical_id'] not in allowed: continue
            assert item['split']=='train' and original[item['parent_component']]=='train'
            out[item['canonical_id']]=item
        assert set(out)==set(allowed);return out
    captures=rows('captures_final.jsonl');routes=rows('video_routes.jsonl')
    for cid,c in allowed.items():
        assert c['probe_split']==split['parents'][c['parent_component']]
        assert captures[cid]['parent_component']==c['parent_component']
        assert captures[cid]['train_ready_p1'] and routes[cid]['all_available']
        assert all(c[k]==captures[cid][k] for k in ['numeric_cache','window_cache'])
    return split,captures,routes
if __name__=='__main__':
    s,c,r=contracts();record={'timestamp_brt':brt(),'split_sha256':sha(R/'split/probe_split.json'),'parents':len(s['parents']),'captures':len(c),'all_source_parents_train':True,'dev_test_samples_read':0,'source_revision':3,'split_seed':s['seed']}
    atomic(P/'split_audit.json',record);update(split_audit_complete=True,status='waiting_compute')
    append_report('## 划分独立复核\n\n'+json.dumps(record,ensure_ascii=False)+'\n\nF划分seed=20261112084；本会话采样/训练seed=20261112。F锁定划分是唯一权威，不重划。仅对允许canonical_id开启缓存/视频，所有dev/test数值与图像均不读。')
    print(json.dumps(record))
