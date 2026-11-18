"""Read 16 bytes only; no modifications, datasets, or model operations."""
import json,concurrent.futures
from .common import *
def main():
    p=json.loads((C/'artifacts/cache_plan_readable_v2.json').read_text());todo=[(c['canonical_id'],cam['camera'],cam['path']) for c in p['captures'] for cam in c['cameras']]
    def check(row):
        try:
            with open(row[2],'rb') as f:head=f.read(16)
            if not head:raise IOError('empty video')
            return None
        except OSError as e:return dict(canonical_id=row[0],camera=row[1],path=row[2],reason=str(e))
    bad=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(check,todo)):
            if r:bad.append(r)
            if i%100==0:atomic(C/'artifacts/source_readability_status.json',dict(status='checking',checked=i,total=len(todo),failures=len(bad),updated_brt=brt()))
    atomic(C/'artifacts/source_readability_v2.json',dict(status='all_remaining_paths_readable' if not bad else 'additional_source_gaps',checked_camera_paths=len(todo),failures=bad,timestamp_brt=brt()));print('checked',len(todo),'bad',len(bad),flush=True)
if __name__=='__main__':main()
