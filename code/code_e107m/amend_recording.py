from e103d_runtime import *
import ast,shutil
assert json.loads((CONTROL/'PIPELINE.json').read_text())['phase']=='e107m_train_M1'
assert not (CONTROL/'eval_BASE').exists()
lock=json.loads((CONTROL/'PRETRAIN_LOCK.json').read_text());changes={}
for name in ('eval','analyze'):
 p=ROOT/f'e107m_{name}.py';q=ROOT/f'e107m_{name}_complete.py';ast.parse(q.read_text());old=CONTROL/f'e107m_{name}_locked_initial.py';assert not old.exists();shutil.copyfile(p,old);p.write_text(q.read_text());changes[str(p.relative_to(ROOT))]=dict(before=lock['source_sha256'][p.name],after=sha(p),snapshot=str(old));lock['source_sha256'][p.name]=sha(p)
lock['recording_correction']=dict(brt=brt(),before_any_dev_forward=True,changes=changes,reason='Save existing K4 full-token outputs and apply preregistered E107P variance definition to sample population; no training, seed, score or criterion change')
atomic(CONTROL/'PRETRAIN_LOCK.json',lock);report('## 评估记录补全（任何dev前向前）\n'+json.dumps(lock['recording_correction'],ensure_ascii=False)+'\n只补齐原已预注册的样本分布方差比所需full-token保存，K4样本/噪声完全相同，不增加模型前向；每parent均匀窗、窗内K4均匀，复制真实Y的权重亦除K，按原summarize_curve重权。原锁源码永久保留；训练文件与所有判据未改。\n')
print('recording correction locked before evaluation')
