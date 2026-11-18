from e103d_runtime import *
lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());bad=[]
for p,h in lock.get('input_hashes',{}).items():
 if not Path(p).is_file() or sha(Path(p))!=h:bad.append(p)
out=dict(checked=len(lock.get('input_hashes',{})),changed=bad,brt=brt());atomic(CONTROL/'INHERITED_INPUT_AUDIT.json',out);assert not bad,bad
report('## 继承输入只读核验\n'+json.dumps(out,ensure_ascii=False));print(json.dumps(out,ensure_ascii=False))
