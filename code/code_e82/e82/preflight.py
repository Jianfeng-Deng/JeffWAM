"""Formal launch gate. Read-only; profile mode never bypasses into formal training."""
import argparse,json
from pathlib import Path
from .config import Config
from .data import read_ready,DataPackageDataset,StatContract,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stats',required=True);p.add_argument('--s0',required=True);p.add_argument('--profile',required=True);a=p.parse_args();c=Config.load(a.config)
    read_ready(c.data_root);ds=DataPackageDataset(c.data_root,'P3_joint');stats=StatContract.load(a.stats)
    if stats.d.get('fixture') or stats.d.get('data_refs')!=ds.refs:raise ValueError('Frozen real stat contract required')
    if stats.d.get('encoder_sha256')!=sha256(c.dino_weights):raise ValueError('Encoder hash changed')
    if not Path(a.s0).is_file():raise FileNotFoundError(a.s0)
    profile=json.loads(Path(a.profile).read_text())
    if profile.get('config_hash')!=c.digest():raise ValueError('Profile config changed')
    if profile.get('fixture') or not profile.get('within48') or profile.get('status')!='complete':raise ValueError('Real GPU profile must fit paired 48 GPUh budget')
    for stage,e in profile['evidence'].items():
        if e.get('refs')!=ds.refs:raise ValueError('Profile predates current data lock')
    print(json.dumps(dict(status='passed',config_hash=c.digest(),data_refs=ds.refs,stats_sha256=sha256(a.stats),s0_sha256=sha256(a.s0),profile_sha256=sha256(a.profile)),indent=2))
if __name__=='__main__':main()
