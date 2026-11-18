from .common import *
protected={}
for folder in ('code_e82','code_e86','code_e88','code_e90'):
 for p in sorted((T/folder).rglob('*.py')):
  if '__pycache__' not in p.parts:protected[str(p)]=sha(p)
for p in (T/'notes/e93_dev_diag_20261114.md',T/'notes/e90_full_v31_p1_20261114.md',E93/'dev/windows_full.json',E93/'train64_selection.json'):
 protected[str(p)]=sha(p)
assert not (R/'protected_inputs.json').exists();atomic(R/'protected_inputs.json',protected);print('protected snapshot files',len(protected))
