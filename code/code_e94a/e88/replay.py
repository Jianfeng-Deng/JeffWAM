"""Independent-process exact next-update replay of a bounded smoke checkpoint."""
import sys,json,torch
from e86.data import CappedParentSampler
from .train import make_data,configure,restore,update,digest
from .common import *
from .model import WorldModel
from .resource import Guard

def main(runid,name):
    run=ART/runid;dest=run/(name+'_external_resume.json')
    if dest.exists():raise FileExistsError(dest)
    p,r,stats,full,small,lc=make_data();configure();stage='P1' if name=='P1' else 'P3';language='short' if name=='P1' else name.split('_',1)[1];guard=Guard('E88_external_resume_'+name,180)
    try:
        m=WorldModel().to('cuda:0').train();opt=torch.optim.AdamW(m.parameters(),lr=3e-4 if stage=='P1' else 1e-4,weight_decay=1e-4);sampler=CappedParentSampler(small[stage].rows,seed=88 if stage=='P1' else 89);refs=dict(contract=sha(run/'contract.json'));step=restore(run/(name+'_resume.pt'),m,opt,sampler,refs);v,d=update(m,opt,sampler,small[stage],stage,step,language,guard,recipe=json.loads((run/'contract.json').read_text())['recipe']);dg=digest(m);expected=json.loads((run/(name+'_resume_check.json')).read_text());out=dict(status='passed' if v==expected['loss'] and dg==expected['digest'] else 'failed',loss=v,expected_loss=expected['loss'],parameters_equal=dg==expected['digest'],elapsed_s=guard.check());atomic(dest,out);assert out['status']=='passed'
    finally:guard.finish()
if __name__=='__main__':main(*sys.argv[1:])
