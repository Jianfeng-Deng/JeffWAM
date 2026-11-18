from .common import *
from .data import Package
import json
if __name__=='__main__':
    dest=C/'artifacts';p=Package()
    for stage in ('P1','P3'):
        path=dest/f'windows_{stage}_v2.json';assert not path.exists();atomic(dest/'rebuild_status.json',dict(status='building',stage=stage,updated_brt=brt()))
        ws=p.windows(stage);atomic(path,ws);print(stage,len(ws),flush=True)
    atomic(dest/'READY_windows_v2.json',dict(status='ready',reason='Extend image history to -24/-18/-12/-6 and mask outside original source/approved segment instead of moving anchors',files=[dict(path=str(dest/n),sha256=sha(dest/n)) for n in ('windows_P1_v2.json','windows_P3_v2.json')],refs=p.refs))
    atomic(dest/'rebuild_status.json',dict(status='ready',updated_brt=brt()))
