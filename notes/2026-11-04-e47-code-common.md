# E47 编码共同约定（天机 P0–P1；三个 luna 会话并行，文件互不重叠）

设计合同：notes/2026-11-04-e47-p0p1-design.md（必须遵守，函数签名见 §5，CPU 烟测清单见 §6）。先通读，再读 tools/e32_data.py、e33_model.py、e33_train.py、e33_eval.py、e32_eval.py、tools/e46_project.py、runs/e46_recalib_20261104/params_adopted.json。

## 分工与文件所有权（只写自己的文件，不碰别人的）
- **A（data）**：tools/e47_config.py（配置 schema `e47-p0p1-v1`，含 §9 的全部配置项：robot_id、action_dim、joint_order、camera、fps、horizons、chunk_len、grid、latent_dim、fk 配置、split、stats floors、loss/EMA、预算…，提供 `load_config(path_or_dict)->dict`、`default_tianji_config()`），tools/e47_data.py（§5 全部函数），tools/e47_verify_data.py（§6 第 1–4、10 项中数据相关项）。**请最先写 e47_config.py**，B/C 读取它。
- **B（model+train）**：tools/e47_model.py（§5 model 全部函数，E47 类，编码器/F/D/IDM/flow，rollout，load_run 拒绝 E33/E33sub 隐式加载）、tools/e47_train.py（compute_losses、update_ema、probe_grad_shares、preflight、time_profile、train，完整 RNG/优化器/EMA/PAUSE/续跑）、tools/e47_verify_model.py（§6 第 5–9、11–12 项）。另提供 B-refit 臂（原 E33 结构/损失在 80 集 fit、官方 DINO 初始权重、新 fit stats）的实现开关（可复用 tools/e33_model.py 代码路径但不修改它，通过导入/子类/复制到 e47 文件）。
- **C（eval）**：tools/e47_eval.py（§5 eval 全部函数：replay_e33_compat、evaluate、action_swap、mask_metrics、pixel_metrics、paired_episode_bootstrap、judge；门槛 (a)–(e) 与 §8 统计，含 legacy 广播 bug 复现与 corrected_obj_h）、tools/e47_verify_eval.py（§6 第 10 项及指标手算哨兵）。
- 集成与跨文件冒烟（`tools/e47_verify.py main()` 调用三个子校验）由第二轮完成；本轮各自只保证自己文件的 CPU 烟测通过，并用接口 stub 处理对其它文件的依赖（依赖的函数签名以合同 §5 为准，若发现合同矛盾，写入报告并采用保守解释，不要擅自改合同）。

## 通用约束
- CPU、CUDA_VISIBLE_DEVICES 为空、线程≤8、`python -B`、PYTHONDONTWRITEBYTECODE=1；不用 GPU，不启动任何训练，不下载权重（DINOv2 官方权重沿用 E33 的加载路径，只读）。
- 不修改 tools/e2*、tools/e3[0-9]_*、tools/e46_*、既有 runs/data；colleague-e、colleague-c 严格只读（colleague-c 仅用现存 Python 环境，字节码与缓存重定向到 runs/e47_p0p1_20261104/cache/）。不用腕部相机。
- 所有路径只读打开源数据（mmap 'r'）；新输出只写 runs/e47_p0p1_20261104/；开始/结束记录相关源文件 SHA256（注意 tools/e36b_*.py 正被别的会话修改，不要把它们当保护对象）。
- 报告各自写 runs/e47_p0p1_20261104/IMPL_REPORT_{A,B,C}.md（中文≤40 行：完成项、烟测结果、与合同不一致或需决策项、未完成项、参数量/估算），并追加 docs/experiment_log/EXPERIMENT_LOG.md 与 notes/worklog.md（BRT=宿主机 CST−11h；追加时用原子追加，避免覆盖他人条目）。
