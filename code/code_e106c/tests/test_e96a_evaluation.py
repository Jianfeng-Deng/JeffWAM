import copy
import pytest
import torch
from eval_e96a import paired_delta,agg,enrich

def test_paired_parent_skill_difference_uses_ratio_of_parent_means():
    a=[];b=[]
    for p,c in enumerate((1.,2.,7.)):
        for j in range(p+1):
            r=dict(index=p*10+j,h=10,parent=str(p),model=.7*c,copy=c)
            a.append(r);b.append(dict(r,model=.8*c))
    # Windows are deliberately unbalanced across parents.
    # A constant ten-point gain must remain ten points for every bootstrap draw.
    for h in (25,50,100):
        a.extend([dict(r,h=h) for r in list(a) if r['h']==10]);b.extend([dict(r,h=h) for r in list(b) if r['h']==10])
    x=paired_delta(a,b,'normalized')['10']
    assert x['parents']==3 and x['windows']==6
    assert x['delta_skill']==pytest.approx(.1)
    assert x['ci95']==pytest.approx([.1,.1])
    same=paired_delta(a,a,'normalized')['10'];assert same['delta_skill']==0 and same['ci95']==[0,0]
    with pytest.raises(AssertionError):paired_delta(a,b[:-1],'normalized')

def test_global_averages_raw_vectors_before_squaring():
    current=torch.zeros(1,24,768);truth=torch.zeros_like(current)
    z=truth.reshape(1,3,8,768);z[:,:,::2]=2;z[:,:,1::2]=-1
    b=dict(world_current=current,world_raw=truth)
    r=enrich({},current,b,10)
    assert r['scores']['fine_raw']['copy']==2.5
    assert r['scores']['global_raw']['copy']==.25
    assert r['scores']['global_raw']['model']==.25
