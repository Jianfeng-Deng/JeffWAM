import os,sys,json,hashlib,torch,numpy as np
sys.dont_write_bytecode=True
from e103d_runtime import *
from e99w2_data import load
from e90.data import load_data,sample_horizons
from e90.sampling import MultiSourceSampler
from e96a.train import configure
from e107m_model import StochasticWorld,WorldModel,p3_energy,energy
CK=T/'results/e106c_f2_s90_20261115_02/ckpt_20000.pt'
def digest(m):
 h=hashlib.sha256()
 for n,p in m.named_parameters():
  if not n.startswith('noise_branch.'):h.update(n.encode());h.update(p.detach().numpy().tobytes())
 return h.hexdigest()
def samplers(tr,p1,cp):
 s=MultiSourceSampler(tr.rows,90,{'piper':1.},{},.45);p=MultiSourceSampler(p1.rows,10390,{'piper':1.},{},.45);s.load_state_dict(cp['sampler']);p.load_state_dict(cp['p1sampler']);return s,p
def flow(tr,p1,cp):
 s,p=samplers(tr,p1,cp);h=hashlib.sha256();modes={}
 from e86.objective import mode_for
 for step in range(20001,23001):
  ids=s.sample(32);pids=p.sample(32);hs=sample_horizons(2000+step,90);h.update(json.dumps([ids,pids,list(hs)],default=int).encode());mode=mode_for(2000+step,90,(.25,)*4);modes[mode]=modes.get(mode,0)+1
 return h.hexdigest(),modes

def main():
 configure();torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(90)
 _,_,tr,_=load();_,_,p1,_,_=load_data();cp=torch.load(CK,map_location='cpu',weights_only=False)
 assert cp['step']==20000 and cp['sampler']['count']==640000 and cp['p1sampler']['count']==640000
 for data in (tr,p1):assert all(r['canonical_id']!='piper_4dc107335389f5711a0f65e0' for r in data.rows)
 m=StochasticWorld(tr.norm);missing=m.load_state_dict(cp['model'],strict=False);assert all(k.startswith('noise_branch.') for k in missing.missing_keys) and not missing.unexpected_keys
 original=WorldModel(tr.norm);original.load_state_dict(cp['model']);original.requires_grad_(False);m.eval();original.eval();b=tr.batch([0,1],(10,25,50,100),'short')
 with torch.no_grad():
  o=original.forward_mean(b)['mean'];v=m.forward_mean(b)['mean'];print('z0 diagnostic',torch.isfinite(o).all().item(),torch.isfinite(v).all().item(),(o-v).abs().nan_to_num().max().item(),flush=True);assert torch.isfinite(o).all() and torch.isfinite(v).all();assert torch.equal(o,v);assert torch.equal(m.with_z(v,torch.zeros(2,32)),o)
 before=digest(m);opt=torch.optim.AdamW(m.noise_branch.parameters(),lr=3e-4,weight_decay=1e-4);m.train()
 for step in (22001,22002,22003,22004):
  opt.zero_grad(set_to_none=True);v,_=p3_energy(m,b,step)
  with torch.no_grad():out=m.forward_mean(b)['mean']
  v=v+.1*energy(m,b,out,step);assert torch.isfinite(v);v.backward();opt.step()
 assert digest(m)==before
 a,counts=flow(tr,p1,cp);print('first full flow',a,flush=True);bb,counts2=flow(tr,p1,cp);assert a==bb and counts==counts2
 lock=json.loads((T/'results/e106c_control_20261115_02/PRETRAIN_LOCK.json').read_text());lock.update(resume_sha=sha(CK),checkpoints={'BASE':str(CK),'M0':str(T/'results/e107m_m0_s90_20261116_01/ckpt_23000.pt'),'M1':str(T/'results/e107m_m1_s90_20261116_01/ckpt_23000.pt')},N=3000,noise_dim=32,noise_hidden=256,K=4,mean_weight=.001,energy_distance='joint horizon flattened standardized tokens Euclidean RMS; V-statistic including diagonal',flow_sha=a,modes=counts,brt=brt(),preregistered='all four conditions and horizons; primary no_traj, confirmatory gt_action; no_traj energy and CRPS improvement >=2 horizons lowerCI>0, mean skill all horizons lowerCI>-.01, spread/skill .8..1.2 each horizon; both k25/k50 openloop mean-vs-sample error improvement lowerCI>0; otherwise no M2/no default change')
 atomic(CONTROL/'PRETRAIN_LOCK.json',lock);atomic(CONTROL/'CPU_TESTS.json',dict(status='passed',z0_bitwise=True,frozen_before=before,frozen_after=digest(m),same_flow_sha=a,modes=counts,CUDA_initialized=torch.cuda.is_initialized(),brt=brt()));assert not torch.cuda.is_initialized()
 report('## 实现锁与CPU单元测试通过\n'+json.dumps({k:lock[k] for k in ('N','noise_dim','noise_hidden','K','mean_weight','energy_distance','flow_sha','modes','preregistered')},ensure_ascii=False)+'\n原头z=0逐位一致；模拟4次更新后所有旧参数SHA逐位不变；从checkpoint恢复的两路完整3000步P3/P1样本+时域哈希相同。M1分支仅替换世界回归监督；冻结主干/动作/原均值头，cos/adapter/alignment/旧链为冻结辅助项，不对分支产生梯度。P1辅助权重仍0.1，P3世界权重仍0.1。新分支AdamW矩从零开始，旧参数不更新；M0保留完整原优化器。\n')
 print('ALL CPU TESTS PASSED',flush=True)
if __name__=='__main__':main()
