"""Evaluation-only recovery after missing results parent; no training updates."""
import json,traceback
from pathlib import Path
import torch
from e90.common import *
from e90.train import configure
from e90.data import load_data
from e90.model import WorldModel
from e90.evaluate import evaluate
from e90.resource import Guard

runid='e90_p1_20261114_01';run=ART/runid;guard=None
previous=json.loads((run/'status.json').read_text())
if previous['status']!='stopped':raise ValueError('Unexpected prior status')
archive=run/'training_driver_stopped.json'
if archive.exists():raise FileExistsError(archive)
atomic(archive,previous)
summary=json.loads((run/'train_summary.json').read_text())
assert summary['steps']==5000 and summary['code_hash']==source_hash()
assert sha(summary['checkpoint'])==summary['checkpoint_sha256']
result=T/'results'/runid
result.parent.mkdir(exist_ok=True)
result.mkdir(exist_ok=False)
try:
    _,_,_,dev,_=load_data();configure();guard=Guard('E90_ckpt5000_eval_tail',600)
    m=WorldModel().to('cuda:0');d=torch.load(summary['checkpoint'],map_location='cpu',weights_only=False)
    assert d['step']==5000 and d['code_hash']==source_hash()
    m.load_state_dict(d['model']);atomic(result/'checkpoint.json',summary)
    evaluate(m,dev,guard,result/'ckpt1_dev_eval.json',steps=4)
    atomic(run/'eval_tail_status.json',dict(status='completed',checkpoint=summary['checkpoint'],checkpoint_sha256=summary['checkpoint_sha256'],elapsed_s=guard.check(),no_training_updates=True))
    atomic(run/'status.json',dict(status='completed',step=5000,peak_allocated_mib=max(torch.cuda.max_memory_allocated()/2**20,1082.28515625),brt=brt(),evaluation_recovered=True,prior_driver_status=str(archive)))
    report('## 5000步与评估入口收尾修复\n5000步与10份checkpoint已完成。自动评估创建结果目录时因T/results父目录不存在而退出（训练权重未丢失）；原错误保留于training_driver_stopped.json与p1_01.log，该段GPU墙钟已全计入。补建本用户T/results父目录，在eval_tail_e90.py只读加载同一ckpt_05000.pt，未更新任何参数，固定4Euler完成同一dev评估。训练包未改、代码hash不变；完整评估见results/'+runid+'/ckpt1_dev_eval.json。')
except BaseException as e:
    atomic(run/'eval_tail_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
finally:
    if guard:guard.finish()
