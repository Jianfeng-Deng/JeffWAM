# E86-M Piper v3

仅 Piper；不修改 `code_e82`，不启动正式训练。所有数据和权重原位读取。服务器使用 `/mnt/workspace/dengjianfeng/venvs/e29/bin/python`，设置 HF/transformers offline，不安装或下载任何依赖。

模型：本地冻结 SigLIP2 base-patch16-224 双塔；RGB 全视野保比例 bilinear 缩至 224、居中 edge-pad、(x/255−.5)/.5，padding 区域按面积排除。冻结 patch 14×14 先固定池化到2×4×768，fp16 落盘再转 fp32 使用。与 E84 同预处理，与官方方形拉伸有差异。E84 绝大多数只有global特征，不满足空间token合同，未当作完整新缓存复用。

当前三路每路 learned attention pooling 8 token，共24；四个历史时刻(-.8/-.6/-.4/-.2s)先加各相机latent ID，再每帧三路合并 learned 8，共32；文本每个≤64 token 的完整句组编码一个768维token，带NULL槽。O=57+文本句组数。A=1本体状态+50带噪动作。W=3未来×3相机×8空间token=72，原生768维冻结目标，无可训目标压缩。O只读O；A/W双向读O/A/W。共享6×256，8头，FFN1024，Pre-LN。动作12关节相对当前q，双夹爪signed absolute (m)，不裁负值；保持既有 next-frame measured feedback 合同，不称发送命令。

主模式 30%同步、30%独立、30%删除W、10%干净动作；默认在同步/独立有W更新中25%概率增加干净动作世界辅助调用。该25%与原任务比例冲突是本轮显式暂定解释，记录于run contract，用户未确认前不可当正式配方。干净模式另算一次action_only动作损失；每个动作样本恰好一个主要LA。世界FM+0.1确定性余弦，P3世界权重0.1；对齐0.01，训练中的FK与E83只是目标，不作任何测试真值/阈值/选型依据。历史整段15%丢弃，状态噪声0.02标准差（训练尺度）/10%丢弃。quality_condition默认false；quality_weight不提供即等权。

数据入口只认固定哈希的rev4两级READY和P3_ready_V1；V1源hash常量固定，206条/24477窗口，不读取泛化candidate作P3。parent全表核对，所有继承numeric/window依赖有发布hash；不重划、不跨缝、不平移动作或标签。历史不足-24时屏蔽而不移动anchor。P3源mask/批准子区间外历史屏蔽。demo3 sampler按实际抽到的窗口累计数硬≤45%，队列恢复同样保持上限，全局parent等权优先，只有超45%时否决demo3候选。

训练数据统计由P3 train parent均衡、最多每parent8窗拟合；动作每offset/维均值与std，关节floor.01rad，夹爪floor.001m；P1→P3保持同一尺度。世界使用固定SigLIP2原始特征，不拟合目标投影或dev尺度。未来端点固定+10/+25/+50。状态等待保留可得时间，部署接口实际等待后才调用。

CPU合同（真实编码器也只CPU）：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/mnt/workspace/dengjianfeng/venvs/e29/bin/python -m pytest -q
```

本轮后台链：

```bash
setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python -u -m e86.driver \
  > artifacts/driver_NEW.log 2>&1 < /dev/null &
```

`driver.py` 中run ID、输出目录固定v1，仅是本次审计命令；原位重跑会拒绝既有路径。任何续跑须新ID/路径或显式读取已有已完成receipt，不覆盖旧结果。driver只等待本会话前置，再提取小dev缓存、P1 192步×8、P3 384步×8和同P1初始化无语言384步×8、batch32每阶段100更新计时、两种模式各200调用时延；没有正式训练launcher。每阶段独立AdamW，5%warmup/cosine，clip1；固定stage-end，不选best。checkpoint存optimizer、sampler、CPU/本卡CUDA RNG、Python/NumPy RNG、配置和数据/代码哈希，恢复拒绝不一致；CPU不调用CUDA RNG。实际烟雾中重放下一更新，报告精确参数差。

GPU Guard在导入CUDA模型前查询nvidia-smi，单卡free≥估算峰值16GB×1.5，选最少利用率；不修改或终止他人进程、不用MPS。显存余量<4GB、OOM、阶段/全局预算超限立即停止。所有CUDA阶段全进程墙钟（包括等待I/O）累计≤3GPUh，跨进程共用带独占文件锁的账本。活动段异常遗留时拒绝新段，须先核账，不会静默归零。

dev固定6 parent×8窗，纯真实反馈真值，不产生模型语言。沿用原始人工短指令作通用条件；比较copy-current、action_only、joint、推理时去语言，以及相同P1初始化/数据/RNG/步数的P3无语言训练。前8/H50报告parent等权关节L1(rad/deg)、夹爪L1(mm)。仅烟雾参考，历史dev不是新盲测集。

时延：`python -m e86.eval_cli smoke_NEW`，每模式8预热+200记分；4噪声×10Euler，完整路径包括三路当前视频原位解码、CPU预处理、GPU冻结编码、缓存历史/原人工指令、回放真实状态等待、策略集成；另记缓存特征策略时间。离线回放不是实机调度或控制验收。正式GPUh用batch32实测均步耗时按P1 5000/P3 10000外推，并另报25%余量与配对R/P场景，缓存重建和排队另计。

源权限例外：demo3 `piper_4dc107335389f5711a0f65e0` 的左腕两份别名均 Permission denied。`artifacts/cache_plan_readable_v2.json` 显式隔离该整片的1,794个P1窗口，rev4原108,851窗保持只读，可读P1=107,057；冻结P3 V1不受影响。`features_v3` 复用185个完成分片，核验旧合同（encoder/preprocess）、receipt、逐文件hash、源路径、parent/split、row列表后建立只读符号链接；停止的v1/v2证据保留。该源权限恢复或批准两视角mask之前，不宣称全部rev4 P1缓存完成。当前控制器状态 `driver_v2_status.json`；原v1因权限缺口停止，未启动训练。

最终历史可得性全表审计使用 `READY_windows_v3.json`：三个池新增mask均0，原窗口hash不变。最终55项CPU测试的发布日志为 `contracts_release_v8.log`；最终可训练参数6,830,350，另375,187,970冻结编码器参数。独立 `e86.watchdog` 只检查账本中的E86进程，在phase/总墙钟上限前20秒终止经过uid/cwd/cmdline三重验证的本会话worker，处理底层I/O挂起；不会操作他人进程。

独立恢复入口（新目录，最多额外128步，不能越过本烟雾原192/384步终点）：

```bash
python -m e86.resume artifacts/smoke_v2/P3_resume.pt resume_NEW --extra-steps 1
```

恢复会核对原contract、当前数据/统计/缓存/文本/代码hash，重建相同optimizer/sampler/RNG/LR并与原run同一更新loss逐位对照；不会覆盖原checkpoint或延长为正式训练。

对齐损失只计算实际可用的图像/历史/动作表示：P1不存在的动作、被丢弃/缺失的历史不会获得对应对齐梯度。

文本句组保留完整顺序，使用无长度截断的固定正弦位置编码；交换句组顺序会改变动作输出。固定位置编码不增加可训练参数。

对齐正例只按固定文本向量的精确身份定义；同parent不同运动描述不作正例或负例，防止可训练投影通过合并正例作弊；图像/历史/动作↔文本使用对称对比损失。

采样先全局parent等权、再parent内窗口均匀；demo3硬上限按累计实际窗口前缀执行，仅在候选会超限时重抽非demo3。避免以50/50 cohort采样额外抬高demo3 parent权重。

GPU调试记录：smoke_v1完成P1后，在P3首次辅助分支暴露随机分配遗漏device参数，已停止并保留原结果。修复后55项CPU回归及全部4模式强制辅助GPU预检通过，使用新smoke_v2从S0重跑；未绕过旧checkpoint代码hash。当前控制器为scripts/continue_smoke_v2.py，状态driver_v3_status.json；所有失败/重跑计入同一3GPUh。
