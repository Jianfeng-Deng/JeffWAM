import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,pathlib,traceback
from common import *
from contract import sha
root=R/'features'/RUN/'frozen';ready=json.loads((root/'READY_features.json').read_text());verified=[];source_count=0
try:
    atomic(P/'feature_verification_status.json',{'timestamp_brt':brt(),'pid':os.getpid(),'status':'running'})
    for i,p in enumerate(sorted(root.glob('piper_*_receipt.json'))):
        r=json.loads(p.read_text());q=pathlib.Path(r['path']);assert sha(q)==r['sha256'];assert r['source_stat_before_after_equal']
        for src,expected in r['sources'].items():
            s=pathlib.Path(src).stat();actual=[s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino];assert actual==expected,(src,actual,expected);source_count+=1
        s=q.stat();verified.append({'path':str(q),'sha256':r['sha256'],'bytes':s.st_size,'stat':[s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino],'receipt_sha256':sha(p)})
        if i%50==0:atomic(P/'feature_verification_status.json',{'timestamp_brt':brt(),'pid':os.getpid(),'status':'running','verified':len(verified),'expected':ready['captures']})
    assert len(verified)==ready['captures']==542
    atomic(P/'READY_feature_verification.json',{'timestamp_brt':brt(),'status':'completed','feature_ready_sha256':sha(root/'READY_features.json'),'captures':len(verified),'source_stats_rechecked':source_count,'source_stat_changes':0,'features':verified})
    atomic(P/'feature_verification_status.json',{'timestamp_brt':brt(),'pid':os.getpid(),'status':'completed','verified':len(verified),'source_stats_rechecked':source_count})
    update(feature_sha_verification_complete=True,source_stat_changes=0)
    print('FEATURE_HASH_AND_SOURCE_STAT_VERIFIED',len(verified),source_count,flush=True)
except BaseException:
    atomic(P/'feature_verification_status.json',{'timestamp_brt':brt(),'status':'failed','traceback':traceback.format_exc()});raise
