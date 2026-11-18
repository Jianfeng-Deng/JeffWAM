import json,copy
from pathlib import Path
import numpy as np,pytest,torch
from e86.common import C,sha
from e86.data import Package,verify_receipt,verify_entries,validate_interval,CappedParentSampler,CachedDataset,build_window
from e86.v3_contract import *
from e86.objective import mode_for
A=C/'artifacts/E86-V3-20261113T144520'
@pytest.fixture(scope='module')
def p3():return Package()
def test_v3_default_receipt_and_entries(p3):
 assert p3.version=='V3' and sha(p3.receipt_path)==V3_RECEIPT_HASH
 assert len(p3.entries)==284 and sum(e['windows'] for e in p3.entries.values())==34777
 assert len(p3.by_canonical)==230 and len({e['parent_id'] for e in p3.entries.values()})==145
 assert sum(e.get('origin')=='E82_Q3_new_accept_cycle' for e in p3.entries.values())==78
 assert all(p3.split[e['parent_id']]=='train' for e in p3.entries.values())
def test_v3_ready_tamper_rejected(tmp_path,p3):
 fake=tmp_path/'P3_ready_V3.json';d=copy.deepcopy(p3.receipt);d['statistics']['windows']+=1;fake.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='Hash mismatch'):verify_receipt(fake,V3_RECEIPT_HASH)
def test_v3_entry_tamper_rejected(p3):
 d=copy.deepcopy(p3.p3);d['entries'][0]['window_anchor_raw_rows'][0]+=1
 with pytest.raises(ValueError,match='Entry hash'):verify_entries(d)
@pytest.mark.parametrize('bad',('lower_anchor','upper_anchor','obs_camera','world_camera','action'))
def test_interval_rejects_crossings(bad):
 a=16;z={'original_action_index':np.arange(1,100)};e={'subinterval_raw_rows_exclusive':[10,67]};o=np.full((2,3),16);w=np.full((3,3),66)
 validate_interval(z,a,o,w,e)
 if bad=='lower_anchor':e['subinterval_raw_rows_exclusive'][0]=11
 if bad=='upper_anchor':e['subinterval_raw_rows_exclusive'][1]=66
 if bad=='obs_camera':o[0,1]=9
 if bad=='world_camera':w[-1,2]=67
 if bad=='action':z['original_action_index'][a+49]=67
 with pytest.raises(ValueError):validate_interval(z,a,o,w,e)
def test_v3_windows_full_membership_interval_and_exclusion(p3):
 ws=json.loads((A/'windows_P3.json').read_text());w1=json.loads((A/'windows_P1.json').read_text());assert len(ws)==34777 and len(w1)==107057
 assert all(r['canonical_id']!=EXCLUDED_CANONICAL and r['split']=='train' for r in ws+w1)
 assert len({(r['canonical_id'],r['anchor']) for r in ws})==34777
 for r in ws:
  e=p3.entries[r['entry_id']];lo,hi=e['subinterval_raw_rows_exclusive'];a=r['anchor'];assert lo<=a-6 and a+50<hi
  assert a in e['window_anchor_raw_rows']
  for grid in (r['obs_rows'],r['future_rows']):
   assert all(x<0 or lo<=x<hi for ids in grid for x in ids)
def test_permanent_exclusion_blocks_dataset_and_sampler():
 row=dict(canonical_id=EXCLUDED_CANONICAL,parent_component='p',split='train',demo3=True)
 with pytest.raises(ValueError,match='excluded'):CachedDataset([row],[],'.',{})
 with pytest.raises(ValueError,match='Ineligible'):CappedParentSampler([row])
@pytest.mark.parametrize('strategy',('parent_equal','stratified'))
def test_v3_sampler_cap_resume_and_train_only(strategy):
 rs=[dict(parent_component=str(i),demo3=i<90,stratum='demo3' if i<90 else 'demo6native',split='train') for i in range(100)]
 a=CappedParentSampler(rs,seed=863,cap=.33,strategy=strategy);a.sample(177);state=a.state_dict();out=a.sample(10000)
 assert a.demo3_count/a.count<=.33
 b=CappedParentSampler(rs,seed=863,cap=.33,strategy=strategy);b.load_state_dict(state);assert b.sample(10000)==out
 with pytest.raises(ValueError):CappedParentSampler([dict(rs[-1],split='dev')],strategy=strategy)
 with pytest.raises(ValueError):CappedParentSampler(rs,cap=.451,strategy=strategy)
def test_stratified_sampling_weights():
 rs=[dict(parent_component=str(i),demo3=False,stratum='a' if i==0 else 'b') for i in range(21)]
 s=CappedParentSampler(rs,seed=3,strategy='stratified',stratum_weights={'a':.25,'b':.75});ids=s.sample(10000);share=sum(i==0 for i in ids)/len(ids);assert .23<share<.27
def test_mode_probability_config_and_legacy_default():
 assert all(mode_for(i,probabilities=(0,0,0,1))=='clean' for i in range(20))
 with pytest.raises(ValueError):mode_for(0,probabilities=(.3,.3,.3,.25))
 assert abs(sum(mode_for(i)=='clean' for i in range(10000))/10000-.1)<.02
def test_v3_all_required_frames_cached_and_contract(p3):
 d=json.loads((A/'cache_ready_audit.json').read_text());assert d['missing_camera_frames']==0 and d['estimated_incremental_bytes']<=30_000_000_000
 from e86.vision import PREPROCESS
 assert json.loads((C/'artifacts/features_v3/contract.json').read_text())['preprocess']==PREPROCESS
 assert d['test_source_reads']==0
def test_increment_cache_overlay_exact_values_and_duplicate_rejection(tmp_path):
 from e86.data import CAMS
 base=tmp_path/'base';inc=tmp_path/'inc';base.mkdir();inc.mkdir();(base/'contract.json').write_text('{}')
 (inc/'overlay.json').write_text(json.dumps(dict(base_cache=str(base),base_contract_sha256=sha(base/'contract.json'))))
 for root,row,val in ((base,10,1.25),(inc,11,2.5)):
  d=root/'c';d.mkdir();(d/'receipt.json').write_text(json.dumps({'cameras':[{'camera':cam} for cam in CAMS]}));np.save(d/'cam_high_rows.npy',[row]);np.save(d/'cam_high.npy',np.full((1,8,768),val,np.float16))
 ds=CachedDataset([],[],inc,{});v=ds.features('c',0,[10,11,-1]);assert np.array_equal(v[:,0,0],[1.25,2.5,0])
 np.save(inc/'c/cam_high_rows.npy',[10]);ds=CachedDataset([],[],inc,{})
 with pytest.raises(ValueError,match='Duplicate'):ds.features('c',0,[10])
def test_configured_clean_mode_reaches_training_objective():
 from e86.model import WAM
 from e86.fixtures import batch
 from e86.objective import loss
 m=WAM().eval();value,metrics=loss(m,batch(2),'P3',0,mode_probabilities=(0,0,0,1),clean_aux_probability=0)
 assert torch.isfinite(value) and metrics['mode']=='clean' and not metrics['clean_aux']
def test_native_demo6_padding_and_fixed_target_constant():
 from e86.vision import preprocess,spatial_grid
 x,valid,affine=preprocess(np.full((720,1280,3),127,np.uint8))
 assert x.shape==(3,224,224) and valid.sum()==224*126
 assert valid[:49].sum()==0 and valid[49:175].all() and valid[175:].sum()==0
 assert np.allclose(affine,[[.175,0,0],[0,.175,49],[0,0,1]])
 assert torch.allclose(spatial_grid(torch.ones(1,196,768),valid[None]),torch.ones(1,8,768))
