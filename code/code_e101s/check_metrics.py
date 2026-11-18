from run import *
# A predicted boundary is never reused, even for nearby repeated transition types.
pp=[(20,0,1)];tt=[(10,0,1),(30,0,1)]
assert len(match(pp,tt,15))==1
assert len(match([(26,0,1)],[(10,0,1)],15))==0
assert len(match([(40,0,1)],[(10,0,1)],30))==1
assert len(match([(10,0,2)],[(10,0,1)],30))==0
# Unknown reference gap does not create a truth boundary; terminal prediction counts as error.
y=np.array([0,0,-1,1,1]);p=np.array([0,0,10,10,10]);r=record(p,y);a=aggregate([r]);assert r['true']==0 and a['accuracy']==.5
write('metric_contracts.json',dict(one_to_one=True,inclusive_tolerance=True,exact_type=True,unknown_gap_excluded=True,terminal_error=True,brt=now()))
print('metric contracts passed')
