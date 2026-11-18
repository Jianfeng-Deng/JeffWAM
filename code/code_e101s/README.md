# E101-S

冻结研究结果：../results/e101s_20261115_01/，报告：../notes/e101s_stage_controller_20261115.md。

- run.py fit：60-parent 5折CPU谓词头与训练调参，已有锁定产物时拒绝重跑。
- evaluate.py：dev一次评估，DEV_ONCE_STARTED.json存在即拒绝重跑。原评估在制图步骤报错；finish_frozen.py仅从冻结统计/序列恢复后处理，不调用dev控制器。
- online_controller.py：锁定S2+b的CPU逐帧接口；输入为E99-B3 b的21维当前帧谓词概率，顺序与旧spec一致，P073索引13固定弃权。每个新parent/cycle的初始化策略应自行定义；当前reset固定S01，截断片段可能失败。

```python
from online_controller import OnlineStageController
c = OnlineStageController('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/results/e101s_20261115_01')
result = c.step(prob21)  # 当前帧概率，不能输入未来分数或标签
print(result['stage'])
```

因果研究候选，未达到recall@30>=0.7，不是实机部署许可。dev参考标签来自模型，非人工。仅12/60训练parent具有完整袋外轨迹用于控制器调参，60条标注点均有袋外分数。普通HMM自环/最短驻留不等于显式HSMM。S3为非因果附录。
