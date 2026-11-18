from run import *
from online_controller import OnlineStageController
lock=json.loads((R/'CONTROLLER_LOCK.json').read_text());c=OnlineStageController(R);assert PIDS[13]=='P073'
s=full[0];p=np.load(R/(s['parent']+'_b_oof.npz'))['prob'];expected=hmm(emission(p,c.aux),c.aux,**lock['selected']['S2']['config']);actual=np.array([SI[c.step(row)['stage']] for row in p]);assert np.array_equal(actual,expected)
c.reset();assert np.array_equal(actual[:300],np.array([SI[c.step(row)['stage']] for row in p[:300]]))
write('online_interface_contract.json',dict(parent=s['parent'],frames=len(p),batch_online_exact=True,reset_prefix_exact=True,dev_rerun=False))
print('online/batch exact on train frames:',len(p))
