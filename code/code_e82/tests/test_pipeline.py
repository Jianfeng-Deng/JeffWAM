import json
import pytest
import torch
from e82.data import read_ready,sha256,StatContract
from e82.model import PiperWAM
from e82.data import synthetic_batch
from e82.objective import objective
from e82.randomness import Streams
from e82.train import optimizer
from e82.diagnostics import stoploss_monitor


def test_ready_rejects_invalid_status_and_bad_hash(tmp_path):
    ready=tmp_path/'READY_parent_split.json';ready.write_text(json.dumps(dict(status='invalidated',files=[])))
    with pytest.raises(ValueError):read_ready(tmp_path,False)
    source=tmp_path/'small.json';source.write_text('{}')
    ready.write_text(json.dumps(dict(status='parent_split_locked',files=[dict(path=str(source),sha256=sha256(source))])))
    assert read_ready(tmp_path,False)
    source.write_text('changed')
    with pytest.raises(ValueError):read_ready(tmp_path,False)


def test_P1_keeps_entire_A_module_S0():
    torch.set_num_threads(2);m=PiperWAM(fixture=True);m.set_stage('P1');m.train();opt=optimizer(m,3e-4)
    tensors=[p for mod in [m.a_proj,m.action_head,m.q_proj,m.dq_proj,m.hist_proj] for p in mod.parameters()]+[m.types[1]]
    before=[p.clone() for p in tensors]
    loss,_=objective(m,synthetic_batch(),StatContract.fixture(),Streams(1,'P1',0),0,'P1');loss.backward();opt.step()
    for a,b in zip(before,tensors):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_stoploss_uses_only_independent_truth():
    with pytest.raises(ValueError):stoploss_monitor(dict(truth_source='model'))
    assert stoploss_monitor(dict(truth_source='recorded_feedback',aligned_points=[dict(joint_step=1000,P_over_R_K8=1.06),dict(joint_step=2000,P_over_R_K8=1.07)]))
    assert stoploss_monitor(dict(truth_source='human',aligned_points=[])) is None


def test_world_accumulation_denominator_matches_effective_batch():
    # Four modes contain three LW-supervised samples; all four receive one LA.
    B=4; micro=1; counts=[1,1,0,1]; values=[2.,3.,0.,7.]
    accumulated=sum((B/micro*n/3)*v/(B/micro) for n,v in zip(counts,values))
    assert accumulated==pytest.approx((2+3+7)/3)


def test_E82D_native_reader_next_feedback_and_real_time(tmp_path):
    import numpy as np
    import h5py,cv2
    from e82.source import read_D_window
    n=64; i=8; t=np.arange(n)/30.;q=np.arange(n,dtype=np.float32)[:,None]*np.ones((1,14),np.float32)*.001
    raw=q.copy();raw[:,6]=-.002;raw[:,13]=-.003
    source=tmp_path/'episode.hdf5'
    with h5py.File(source,'w') as f:f.create_dataset('observations/qpos',data=q)
    nc=tmp_path/'numeric.npz';ct=np.repeat((t-.001)[:,None],3,1)
    np.savez(nc,timestamp=t,q=q,action=raw[1:],original_action_index=np.arange(1,n),camera_exposure_timestamp=ct,camera_received_timestamp=ct,state_available_time=t+.004,continuous_segment=np.zeros(n,dtype=int))
    paths={}
    for c in ('cam_high','cam_left_wrist','cam_right_wrist'):
        p=tmp_path/(c+'.mp4');wr=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),30.,(84,56))
        if not wr.isOpened():pytest.skip('local OpenCV MP4 encoder unavailable')
        for j in range(n):wr.write(np.full((56,84,3),j*3,dtype=np.uint8))
        wr.release();paths[c]=str(p)
    row=dict(source_path=str(source),numeric_cache=str(nc),anchor=i,decision_time=t[i]+.004,O_video_rows=[[i-6]*3,[i]*3],W_video_rows=[[i+10]*3,[i+25]*3,[i+50]*3],O_age_s=[[.005]*3,[.005]*3],camera_paths=paths,canonical_id='synthetic',parent_component='p',session_id='s',task_id=1)
    b=read_D_window(row,'P3_joint')
    assert b['obs_images'][0].shape==(1,2,3,252,448)
    assert not b['obs_crop_valid'][0,:,2].any() and not b['obs_crop_valid'][0,:,4].any()
    np.testing.assert_array_equal(b['action'][0].numpy(),raw[i+1:i+51])
    np.testing.assert_allclose(b['dq'][0].numpy(),.03,rtol=1e-5)
    assert b['decision_wait_s'].item()==pytest.approx(.004)


def test_world_missing_ROI_preserves_equal_camera_weights():
    from e82.objective import world_loss
    pred=torch.zeros(1,3,91,768); pred[:,:,:25]=1;pred[:,:,25:58]=2;pred[:,:,58:]=3
    valid=torch.ones(1,3,91,dtype=torch.bool);valid[:,:,42:58]=False;valid[:,:,75:91]=False
    assert world_loss(pred,torch.zeros_like(pred),valid).item()==pytest.approx(14/3,rel=1e-6)


def test_structural_missing_W_has_no_latent_condition():
    torch.set_num_threads(2);m=PiperWAM(fixture=True).eval();b=synthetic_batch();o,ov=m.observation(b);a=torch.randn(1,50,14);w=torch.randn(1,3,91,768);v=torch.ones(1,3,91,dtype=torch.bool);v[:,:,42:58]=False;t=torch.zeros(1)
    x=m.predict(o,ov,a,w,t,t,w_valid=v)[0];w[:,:,42:58]+=1000
    y=m.predict(o,ov,a,w,t,t,w_valid=v)[0];torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_deployment_policy_omits_Estar():
    from e82.inference import infer
    m=PiperWAM(fixture=True,inference_only=True).eval(); assert m.vision.target is None
    b=synthetic_batch(); assert torch.isfinite(infer(m,b,StatContract.fixture())).all()
    with pytest.raises(RuntimeError):m.world_target(b)


def test_CPU_checkpoint_never_touches_CUDA_RNG(tmp_path,monkeypatch):
    from e82.train import save,load,optimizer
    def forbidden(*a,**kw):raise AssertionError('CPU checkpoint accessed CUDA')
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',forbidden);monkeypatch.setattr(torch.cuda,'get_rng_state',forbidden);monkeypatch.setattr(torch.cuda,'set_rng_state_all',forbidden);monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    m=PiperWAM(fixture=True);opt=optimizer(m,1e-4);save(tmp_path/'cpu.pt',m,opt,0,{})
    load(tmp_path/'cpu.pt',m,opt,restore_rng=True)


def test_task_default_and_out_of_vocabulary_are_UNK():
    m=PiperWAM(fixture=True).eval();b=synthetic_batch();b.pop('task_id')
    o,v=m.observation(b);b['task_id']=torch.zeros(1,dtype=torch.long);z,_=m.observation(b)
    torch.testing.assert_close(o,z,rtol=0,atol=0)
    b['task_id']=torch.tensor([999]);z,_=m.observation(b);torch.testing.assert_close(o,z,rtol=0,atol=0)
