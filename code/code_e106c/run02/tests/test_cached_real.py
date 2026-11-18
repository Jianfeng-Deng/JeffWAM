import json,pytest,torch,numpy as np
from e86.common import C
from e86.data import Package,CachedDataset,collate
from e86.model import WAM
from e86.objective import loss

def test_real_piper_cached_window_and_next_feedback():
    wsfile=C/'artifacts/windows_P3_v2.json';cache=C/'artifacts/features_v2'
    if not wsfile.exists() or not cache.exists():pytest.skip('Own feature extraction not started')
    ws=json.loads(wsfile.read_text());r=next((r for r in ws if (cache/r['canonical_id']/'receipt.json').exists()),None)
    if r is None:pytest.skip('No completed own P3 capture yet')
    p=Package();stats=json.loads((C/'artifacts/stats_v1.json').read_text());ds=CachedDataset([r],p.captures,cache,stats);sample=ds[0];b=collate([sample]);assert b['obs'].shape==(1,5,3,8,768) and b['world'].shape==(1,72,768)
    cap=next(c for c in p.captures if c['canonical_id']==r['canonical_id']);z=np.load(cap['numeric_cache']);a=r['anchor'];assert np.array_equal(z['original_action_index'][a:a+50],np.arange(a+1,a+51));assert np.array_equal(sample['action'].numpy(),z['action'][a:a+50]);assert sample['decision_wait_s']>=0
    m=WAM().eval();v,d=loss(m,b,'P3',0,mode='independent');assert torch.isfinite(v);v.backward();assert m.blocks[0].ff[0].weight.grad.norm()>0;assert not torch.cuda.is_initialized()
