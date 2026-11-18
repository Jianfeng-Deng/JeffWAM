# E33_sub 全 token 残差重设计（诊断与设计定稿）
- 日期：2026-11-04；结论：主要瓶颈是 240→60 空间池化；采用全部 240×256 token 的真实残差坐标，取消池化/PCA，不采用 token 子集。
- 本次只运行 SSH 内联 CPU 诊断及合成前后向（无 optimizer.step）；CUDA_VISIBLE_DEVICES 置空，Python -B，OMP/MKL/OpenBLAS/NumExpr/torch intra-op≤8、interop=1；CUDA 未初始化，未启动 GPU 或训练任务。
- 远端唯一新增文件为本文；`/mnt/workspace/colleague-e`、`/mnt/workspace/colleague-c` 严格只读，仅借用后者 Python 解释器/依赖；不运行其项目脚本，不写 pycache，不安装软件。
- 遵守根 AGENTS.md：本方案仅隔离的离线研究，不接现役控制器；现役物体规划仍用 MuJoCo。无腕部相机、VLM、教师恢复或上位机操作。

## 1. CPU 消融证据与失败定位
- 输入：`runs/e33sub_20261104/cache_ep80`，fit/dev/eval=2680/132/266；PCA/统计仅拟合 fit，解码诊断用全部132个dev窗口、h=8/16/32，未用官方eval选择表示。
- 固定 B ep80 SHA256=`86595f3d0389b92001da2e0fc7863a0d9833f74f08feb17e12925ea6abe28fe7`；features SHA256=`3e5b1301d22fcc080c17466eabc0aab763afbc3874495be1291a50aa18035502`；index SHA256=`f47b8abeadee11a9ffe727b6940b019b5c040671f3db92b2643860c712c56028`。
- 方法：只从源码 AST 提取 Decoder/MaskHead 两个类，加载 ep80 对应权重；不构建 encoder、不调用 prepare/train、不写缓存。核验 manifest 全部数组哈希，源 checkpoint 哈希匹配。
- PSNR对真未来192×320 RGB逐窗计算；IoU保持原9-slot→左手6/右手7/瓶0–5并集/箱8映射，真值2×2池化后>.5、预测logit>0，空union记1。未更改任何指标定义。
- 令 d=z_h−z_t，P 为12×20→6×10的2×2均值池化，U 为最近邻2倍上采样；pool-only 用 z_t+UP(d)，PCA-only 用 z_t+μ+(d−μ)VVᵀ，组合用 z_t+U[μ+(P(d)−μ)VVᵀ]。
- 表内均为 +16/+32 两数：E=1−Σ‖d−d̂‖²/Σ‖d‖²（dev原始全token能量，%）；ΔP=PSNR_exact−PSNR_reconstructed（dB）；Δ瓶/箱=冻结mask-head的IoU下降；负数代表改善。

| 真实残差重建方案 | E (%) | ΔP | Δ瓶 | Δ箱 | max ΔP / max ΔIoU¹ | 准入 |
|---|---:|---:|---:|---:|---:|---|
| 全240×256，无压缩 | 100/100 | 0/0 | 0/0 | 0/0 | <0.00001 / 0 | 通过 |
| 仅池化60，通道256 | 79.97/80.69 | .881/1.190 | .0518/.0611 | .0044/.0062 | 2.814/1.000 | 失败 |
| 仅PCA64，沿用旧池化fit基 | 87.91/88.58 | .049/.074 | .0003/−.0024 | −.0004/−.0016 | .296/1.000 | 失败 |
| 旧池化60+PCA64 | 71.70/72.82 | .922/1.245 | .0549/.0660 | .0074/.0065 | 2.845/1.000 | 失败 |
| 全240位置，重拟合PCA32 | 75.92/76.34 | .104/.165 | −.0016/−.0052 | .0012/.0026 | .440/1.000 | 失败 |
| 全240位置，重拟合PCA64 | 88.13/88.80 | .042/.065 | .0025/−.0003 | −.0006/−.0011 | .221/.1714 | 失败 |
| 全240位置，重拟合PCA128 | 96.68/96.97 | .009/.012 | .0022/.0033 | .0008/−.0013 | .112/.1500 | 失败 |
| 全240位置，重拟合PCA192 | 99.16/99.25 | .001/.003 | .0003/.0003 | .0009/.0001 | .061/.1500 | 失败 |
| 当前预测前景选128 token | 59.23/61.28 | .176/.338 | 0/0 | 0/0 | 2.677/.6667 | 失败 |
| 当前预测前景选192 token | 80.30/81.58 | .086/.174 | 0/0 | 0/0 | 1.913/.6000 | 失败 |
| oracle运动选128 token | 88.22/88.98 | .023/.043 | −.0000/.0015 | .0018/−.0015 | .163/.3333 | 失败 |
| oracle运动选192 token | 97.56/97.88 | .006/.009 | −.0007/−.0004 | −.0016/−.0006 | .061/.0542 | 失败 |

- ¹max列跨所有132窗、h16/32，IoU还跨左手/右手/瓶/箱；正式准入同现代码继续检查h8/16/32全部，不把均值替代逐窗最大值。上表所有失败方案在h16/32已失败。
- 旧PCA64的90.33%是fit上“池化后中心化通道谱能量”，不是原始全token能量；原PCA32为79.15%。全token重新拟合的fit谱能量32/64/128/192分别为76.24/88.83/97.01/99.26%；不可混用分母。
- 因果token选择：当前冻结mask-head四类概率取max，再3×3邻域max，按分数稳定降序取128/192（平分按空间索引）；两个未来共用当前集合，其余d̂=0。oracle集合按各真实未来‖d_token‖²排序，仅是不可用的诊断上限，不能输入推理。
- 仅池化的PSNR损失已达旧组合的约95.6%；去池化后，同一PCA64基的PSNR损失大幅下降，说明空间平均是主因；PCA仍非无害，离散mask阈值使个别小目标IoU跳变，不能凭均值合格放行。
- 全token解码上限：exact PSNR(+16/+32)=21.5839/21.5629dB，瓶IoU=.44534/.44103，箱=.82145/.81477；这是冻结表征/头的重建上限，不是预测性能或任务成功率。
- 无标准化的z_t+(z_h−z_t)最大token误差1.19e−7、图像误差1.19e−7；fit-only全通道标准化再逆变换最大token误差4.77e−7、图像误差2.98e−7、mask二值翻转0，确认浮点容差内无损。缓存本身为fp16，不把此结论扩大为原始FP32 encoder量化无损。
- 旧PCA64复现与原JSON逐窗PSNR最大差5.15e−5dB（批处理浮点误差）；主消融CPU计时9.02s。受保护的tools/e33*.py、tools/e32*.py及整个已有e33sub运行目录共122文件，前后SHA256完全相同；PREPARED未创建。

## 2. 修订后的表示、生成器与低层接口
- 状态s_h=[r_h,v_h]：r=q[t+h]−q[t]∈R54；v=z[t+h]−z[t]∈R^(240×256)，每时刻61494D，联合pair122988D。空间顺序始终12×20行优先；无P、PCA、白化或目标token丢弃压缩。
- v按h和256通道在fit窗×240位置估计μ_v/σ_v[3,256]，std下限.01；本次fit三个h均未触发下限。r[3,54]、dq[54]统计继续fit-only；dq用因果rad/frame差分，episode首帧零值及valid=0。
- encode_subgoal仍返回[...,61494]，先54D r再flatten视觉；decode_subgoal返回(q_t+反标准化r, z_t+反标准化v)。规范q/action顺序A7+B7+左20+右20，单位rad；未来q只读measured，不以command代替。
- G继续联合R1流/R2确定性：4层pre-LN Transformer decoder、宽192/6头/FFN768/dropout .1，482个query=2×(240视觉+1关节)，cross-attention读取当前240视觉+q/dq两个token（242 memory）；位置改12×20。
- R1噪声/插值/速度监督覆盖完整122988D，同pair共用τ；8步Euler不变。R2使用同482 learned queries直接回归；条件、训练步数、噪声配对协议及低层一致，禁止拆成两个独立未来模型。
- 避免宽192线性投影形成固定192维输出子空间：视觉输出改为共享局部MLP `[LN(h_token),local256,z_t_token]`（704D）→384→SiLU→256；R1 local=xτ对应token，R2 local=0，二者均有相同的当前全通道条件通路。该分支无空间压缩，局部Jacobian可满256秩；禁止另训PCA凑宽度。
- G图像输入投影256→192、关节头192→54；q/v及两时刻损失仍各自取均值再等权。前景损失权重由真值9-slot并集24×40均值池化2×2到12×20，w=.2+.8·占比，逐样本/时刻归一均值1；真值mask只用于训练损失。
- 低层goal_v改256→256，目标位置/null视觉改240位置；action_memory=[当前240、raw当前q、独立dq、目标480、目标q两个]，共724×256。输入为标准化残差token，实际h/模态嵌入保留；绝不把目标q当新观测。
- F、基座predictor/decoder/mask/FK冻结且eval，低层沿用flow/memory/prop初始化与训练白名单，G↔低层detach；仅训练新G/目标适配器与原允许的低层参数，基座文件本身不改。
- 其余训练契约沿用原设计§3–4：pair b16占80%、b8占20%；G60 epochs，低层后40 epochs；AdamW/学习率/warmup/clip/seed32不变；真目标hindsight、50%小噪声(.03)、未来q dropout .25、整组目标dropout .10，不使用生成目标伪动作标签。
- 动作仍32×54、10步flow，b8仅监督前16步；每个指令块始终锚定最新真实q，a=q_t+norm·a_std+a_mean；评测command[k]对应t+k−1，未来实测状态对应t+h；不递归回灌预测q。

## 3. 参数、显存、时间与磁盘
- CPU内存中构建上述G（未改源码）：R1/R2均3,052,918参数；低层按同构扩展精确推算4,619,062，总可训练7,671,980（旧7,093,324，+8.2%）；实施后parameter_counts须逐项复核。
- FP32参数/梯度/Adam两动量约.114GiB；pair的FP32状态batch8约3.75MiB。G每层attention元素从2.13M增至16.75M（batch8/6头），约7.86倍；4层主干每样本MAC从.391G增至1.602G，另计局部头/投影。
- CPU合成batch8单次G前后向（无更新，预热1次后3次均值）：旧.129s，全token R1=.716s、R2=.707s，梯度均有限；这是CPU规模诊断，不能换算H20实测时间或推断训练质量。
- 未测H20估算：缓存训练峰值约3–7GiB，硬上限8GiB；K8推理microbatch1–2约3–6GiB。默认8×accum2=有效16，超显存可改4×4或2×8，保持相同样本/更新/epoch及R1/R2配对；不减少token、不解冻F补救。
- 正式每epoch ceil(2680/16)=168更新，A=3360、B=6720、每运行10080。H20暂估t_A=.15–.25s/update、t_B=.25–.45s/update，单运行.61–1.07 GPUh；两运行加公共评测/保存.3–.6 GPUh合计1.51–2.75 GPUh，均未实测。
- 后续另获训练授权时先用≤.03 GPUh分阶段测速，计入总额；预算公式2(3360t_A+6720t_B)/3600+公共开销，优先≤3 GPUh，含全部试跑/失败/评测绝对上限5.5。超目标先优化缓存/批处理/候选分块，无法兑现则停止报告，不删预注册epoch/运行或自动追加实验。
- 端到端K1估80–320ms、K8分块估140–590ms，存在超原延迟门槛风险；G8步/低层10步不变，条件投影可缓存，K8共享低层noise。后续须预热20次、测200次p50/p95；仍要求K1 p95≤250ms、K8≤500ms，不能宣称15Hz。
- 3450个去重帧已有fp16 token占.395GiB，整个原缓存约.54GiB；直接只读引用，不重提取/复制视频。新stats+准入清单<1MiB；约12份轻量完整checkpoint（含optimizer、RNG）约1.1GiB，指标/固定缩略图预留.3GiB，新增目标≤2GiB、硬限5GiB。
- 不保存全部K8逐步图像/噪声；按window/seed重建。新输出根建议`runs/e33sub_full_20261104/`，manifest引用旧数组哈希；本次不创建目录、READY/PREPARED、checkpoint或launch文件。

## 4. 后续最小改动清单（本次均未实施）
| 文件 | 函数/类与必要修改 |
|---|---|
| tools/e33sub_cache.py | fit_subgoal_stats新增full240模式（μ/σ[3,256]）；encode_subgoal/decode_subgoal直接全token残差；_compression_admission同时记录fit能量、dev原始能量、逐窗PSNR/四IoU；prepare写独立v2侧车，复用旧只读数组，禁覆盖旧stats/PREPARED |
| tools/e33sub_data.py | E33SubData.__getitem__保留现有z/q/action输出，targets变[3,61494]；校验representation_spec/shape/hash，拒绝旧PCA stats，所有memmap继续mode=r |
| tools/e33sub_model.py | JointSubgoalFlow.__init__/_unpack/_queries/_project/loss及velocity/predict/sample扩为240×256和482query，加入local输出分支；E33SubPolicy.__init__/action_memory/infer_candidates改724memory及noise形状；parameter_counts复核 |
| tools/e33sub_train.py | _mask_weight池化改2，compute_losses适配全维；run取消仅8×2硬编码，校验有效batch16；_checkpoint/load_run新增spec、stats/数组hash和维度硬校验，旧v1不得静默加载；保留完整RNG/PAUSE/续跑 |
| tools/e33sub_eval.py | _goal_distance改全240×256且保持q/v等权；_goal_quality/_decode_metrics/_make_goals/_actions_for_goals/evaluate适配新表示，K8分块；_score_actions修正为原预注册的四项“分别除同项hold再平均”，A_mm只作附加，不能用现有raw-mm均值替代主A |
| tools/e33sub_plan.py | candidate_model_inputs继续候选自身f/FK；score_candidates若后续实施，Dv比较两时刻完整标准化残差，当前预测mask权重改240；保留fit中位数标定/动作敏感性门槛，未来真值禁用于选择；当前Phase2 TODO照实保留 |
| tools/e33sub_verify.py | verify替换PCA形状/压缩断言，增加全维往返、局部头满秩通路、724memory、K8分块一致性、规范A评分哨兵；原①–⑫的冻结/因果/无泄漏/续跑/哈希检查全部保留 |
- 新model_spec=`E33Sub-full240-jointpair-v2`，representation_spec=`full240x256-delta-v2`，stats单独版本；source cache-v1只作为不可变数组仓库，不能继承其失败PCA准入状态。tools/e33_*.py、e32_*.py无改动需求，已有runs永久保留。

## 5. 新准入、预注册标准与风险处置
- 表示准入先于任何训练：固定132dev窗、冻结ep80、真实Δz往返；维度完整、能量≥90%、各窗/h8/16/32 PSNR损失≤.5dB、四类IoU下降≤.03；同时发布均值/最大值及前景cosine/LN-L2。保留旧严格max判定，不靠改聚合放行任何备选。
- 新增数值契约：FP32 token往返max≤1e−5、q往返≤1e−6rad、mask二值翻转0；检查σ下限/静态有限值、单样本与末尾batch、空间索引、h嵌入及每模态等权。缓存量化和真实生成质量分别报告，不把无压缩oracle当模型准入。
- 再做原CPU smoke①–⑫：索引/episode边界、measured/command错位哨兵、fit-only统计、固定noise复现、未来信息不影响generated、梯度白名单/G-lo隔离、最新q锚定、候选自身动作、关闭诊断头仍能infer、断点/RNG/PAUSE一致、单真实窗口E2E及旧文件哈希。
- 完整继承原`notes/2026-11-04-e33sub-design.md`§6–7全部预注册标准；主A仍四个h16/32左右EEF误差相对对应hold的均值，按episode等权/配对bootstrap1000次、seed20261103，9个官方eval episode/266窗不变。
- R1/R2都必做、固定最后ep60：generated K1相对B的A降≥5%、相对hold降≥10%、对B差值95%CI严格<0；任一侧/步EEF、旋转、手关节退化≤5%，原数值分母下限不改；无官方eval返调门槛。
- 高层h16/32视觉误差与q RMSE相对hold均改善≥5%且moving也通过；moving仍任一真实EEF位移≥20mm。flow ES较R2降≥5%、best-of8较mean-of-errors降≥10%、均值误差恶化≤5%，否则优先R2，不宣称多模态必要。
- 低层oracle A较null/shuffle低≥10%；generated/oracle动作误差≤1.25，生成目标超fit留episode最近邻95%阈值比例≤10%；去视觉A增加<2%且CI含0仍记未证实视觉价值，oracle未来q收益不得计部署收益。
- 筛选只在f真实动作误差比shuffle低≥10%且差值CI<0后开放；K8-select较first的A降≥3%且CI<0、fallback≤5%、原hold偏置/评分相关/延迟门槛同时过；双向donor方向响应仍≥65% cos>0、中位cos≥.2、有效对≥50，保持原配对约束。
- 风险：61494D目标/时刻对80fit episode样本效率差，flow噪声和背景全局变化可能主导；保留前景加权和模态等权，先看moving与R2，失败即报告，不放宽门槛或增加训练预算。token子集即使oracle192也有IoU最大降.0542，不能作为已获准的节省方案。
- 全token仅消除表示压缩损失，冻结头本身瓶IoU约.44仍有限；生成目标出支持、低层忽略视觉、未来q捷径、f动作不敏感、缺乏任务进度/闭环可达性风险全部保留。只报告离线代理指标；本次结论不包含训练收益、GPU实测或真机成功率。
