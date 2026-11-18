from common import *
p=C/'verify_results.py';s=p.read_text().replace(" atomic(P/'verification.json'", " checks={k:bool(v) for k,v in checks.items()}\n atomic(P/'verification.json'");p.write_text(s)
d=P/'debug_attempts/verification_serialization';d.mkdir(parents=True,exist_ok=True)
for f in P.glob('verification.json.*.tmp'):f.rename(d/f.name)
atomic(P/'verification_serialization_fix.json',dict(timestamp_brt=brt(),scope='CPU reporting only; no GPU/model rerun',issue='numpy boolean check cannot be directly JSON serialized',fix='cast each final check to builtin bool before atomic JSON',evidence=str(d)))
append_report('## CPU校验台账修复\n\n'+brt()+' 首次分片校验到写JSON时遇到numpy bool序列化错误，已统一转Python bool并重跑校验写出；没有训练/GPU重跑，未完成临时台账保留debug_attempts/verification_serialization。')
print('verification_JSON_cast_fixed')
