"""CPU forward evaluation of fixed ckpt-1; no CUDA and no training updates."""
import json,time,traceback
from pathlib import Path
import torch
from e90.common import *
from e90.train import configure
from e90.data import load_data
from e90.model import WorldModel
from evaluate_cpu_tail import evaluate

runid='e90_p1_20261114_01';run=ART/runid;result=T/'results'/runid
if (result/'ckpt1_dev_eval.json').exists():raise FileExistsError('Never overwrite evaluation')
summary=json.loads((run/'train_summary.json').read_text())
assert summary['steps']==5000 and summary['code_hash']==source_hash()
assert sha(summary['checkpoint'])==summary['checkpoint_sha256']
class CPUGuard:
    def __init__(self):self.started=time.monotonic()
    def check(self):
        if torch.cuda.is_initialized():raise RuntimeError('CPU evaluation must not allocate CUDA')
        return time.monotonic()-self.started
try:
    _,_,_,dev,_=load_data();configure();guard=CPUGuard();m=WorldModel()
    d=torch.load(summary['checkpoint'],map_location='cpu',weights_only=False)
    assert d['step']==5000 and d['code_hash']==source_hash();m.load_state_dict(d['model'])
    atomic(result/'checkpoint.json',summary)
    evaluate(m,dev,guard,result/'ckpt1_dev_eval.json',steps=4)
    receipt=dict(status='completed',backend='cpu_float32',elapsed_seconds=guard.check(),GPUh=0,
                 no_training_updates=True,checkpoint_sha256=summary['checkpoint_sha256'],evaluator_sha256=sha(OUT/'evaluate_cpu_tail.py'),
                 future_baseline_rule='Use same CPU evaluator, 4 Euler, same per-window seed and existing frozen rows')
    atomic(run/'eval_cpu_02_status.json',receipt);atomic(result/'evaluation_receipt.json',receipt)
    atomic(run/'status.json',dict(status='completed',step=5000,brt=brt(),evaluation_recovered=True,
                 evaluation_backend='cpu_float32',formal_training_peak_mib=None,formal_peak_guard_bound_mib=8192,
                 prior_driver_status=str(run/'training_driver_stopped.json')))
    report('## 5000步收尾与CPU评估完成\n5000步与10份checkpoint已完成。自动评估因T/results父目录不存在退出；错误保留于training_driver_stopped.json，该GPU段全部计入。首个评估重启在附加util0准入检查被拒，未初始化CUDA、GPUh0。随后eval_cpu_02.py只读加载同一ckpt_05000.pt，用evaluate_cpu_tail.py进行CPU float32评估（仅把GPU设备常量改为CPU，算法/4Euler/窗口和噪声种子不变），无参数更新、无新增缓存。CPU与CUDA随机生成器不同，步骤2比较须复用本CPU evaluator及相同seed；receipt绑定其SHA256。训练包/原checkpoint代码hash未改。完整JSON：results/'+runid+'/ckpt1_dev_eval.json。正式段精确峰值未在目录创建异常前落盘，仅可报告每步guard≤8192MiB，烟雾峰值1082.29MiB。')
except BaseException as e:
    atomic(run/'eval_cpu_02_status.json',dict(status='stopped',error=str(e),traceback=traceback.format_exc()));raise
