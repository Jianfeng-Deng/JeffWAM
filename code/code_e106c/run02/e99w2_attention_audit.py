import sys
sys.dont_write_bytecode=True
import torch,json
from e99w2_common import *
from e99w2_data import load
from e99w2_model import WorldModel
from e96a.train import configure
configure();_,_,tr,_=load();m=WorldModel(tr.norm);m.load_state_dict(torch.load(INIT,map_location='cpu',weights_only=False)['model']);m.eval();b=tr.batch([0],(10,25,50,100),'short');a=torch.zeros(1,50,14);t=torch.ones(1)*.5
with torch.no_grad():
 b=dict(b,fixed_condition=True);x=m(b,a,torch.empty_like(b['world']),t,t);y=m(b,a+1,torch.empty_like(b['world']),t,t);assert torch.equal(x['observation'],y['observation']);short=b['world_horizons'][0]<=50;long=~short;shortdelta=float((x['mean'][:,short]-y['mean'][:,short]).abs().max());longdelta=float((x['mean'][:,long]-y['mean'][:,long]).abs().max());assert shortdelta>1e-6;assert longdelta==0
 m.mean_query.add_(1);z=m(b,a,torch.empty_like(b['world']),t,t);actiondelta=float((z['action']-x['action']).abs().max());assert actiondelta>1e-6
 one=tr.batch([0],(10,),'short');x=m.forward_mean(one)['mean'];changed=one['robot'].clone();changed[:,11:]+=10;y=m.forward_mean(dict(one,robot=changed))['mean'];prefix=float((x-y).abs().max());assert prefix==0
out=dict(status='passed',A_to_short_W_delta=shortdelta,A_to_long_W_direct_token_delta=longdelta,W_to_A_delta=actiondelta,O_A_invariant=True,single_h10_future_after10_perturbation_delta=prefix,torch_CUDA_initialized=torch.cuda.is_initialized(),brt=brt());atomic(ART/'attention_contracts.json',out);report('### O/A/W与轨迹前缀合同\n'+json.dumps(out,ensure_ascii=False)+'\n长W禁止读取A token；动作迭代转成的R计划仍可作为长W条件，这是本轮计划条件的显式路径。no-traj删除整组124D条件（含当前/历史状态与FK），因此其配对收益不单独归因于未来计划。链式训练沿用真实移位观测/状态和计划后缀；本轮部署主评分仅从当前真实观测出发，不声称无反馈的多跳闭环。');print(json.dumps(out))
