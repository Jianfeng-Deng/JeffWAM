# E92b yam uncertain 密集复判

日期 2026-11-14 BRT。状态 进行中。CPU only，GPUh=0。

## 范围与限制
49 个 E92 clip，来源只读，仅训练文本。永久排除指定 Piper episode，冻结评估集不写。v1 保留。复核采用同会话先凭帧记录再对照，独立性较弱，同模型家族，非人工真值。

## 密集抽帧
待完成。

## 逐 clip 判读与因果叙述
待完成。

## 10 clip 复核
待完成。

## 统计与训练建议
待完成。

## 抽帧与浏览完成
49/49 clip，逐 clip 按时间顺序浏览全部顶视密集帧及双腕视。顶视目标间隔0.85秒，上限34帧，加两路腕视各3帧，每clip不超过40帧。图像保留源分辨率，浏览用320×240拼图，拼图只在内存，经SSH显示，无客户端落盘。全部源视频size、mtime、首64KiB SHA256通过E87清单核对，来源split通过训练范围检查。CPU单线程、setsid nohup分离抽帧，GPUh=0。

## 判读完成
29条完整桌面取管片均见单个蓝盖透明管、抬起移向管架、末段松开撤离。20条近架恢复片可见夹爪姿态改变、重新朝向管架并下移，末段夹爪仍贴近管体，保留释放/接触不确定。所有片保留精确洞位与插入深度不确定。后见信息只用于总指令/场景和resolution_notes，过程叙述采用顶视已见状态与同相机前缀证据，避免跨相机同步假设。

## 复核初判已封存
seed=9202抽样10条，先看帧并写e92b_independent_review_before_comparison.json，再构建v2与对照。初判每条独立给出指令、活动臂、4个阶段证据帧及阶段数。此前读过v1报告及前几条文本，同会话记忆无法清除，因此独立性较弱。同模型家族复核，非人工真值。

## 最终统计与复核
完成时间 2026-11-14 04:22:18 BRT。49/49 clip，1556张，单clip25至40张，顶视最大间隔0.9秒。关键帧 41,436,294 字节，本会话数据新增约 42.02 MB，CPU only，GPUh=0。
uncertain 为49/49 → 49/49（100% → 100%），完全消除0条，局部信息获得补充49条。
分类型计数采用多标签，v1由原uncertainty_notes回溯分类。洞位49→49，插入深度49→49，遮挡接触49→49，物体身份0→0，其他29→0。其他29条是目录two_tubes与可见单管范围冲突，v2限定为本clip可见单管，去除第二管动作推断。物体身份原已明确蓝盖透明管，未将其重复计为新增解决。
总指令修改20/49（40.82%），均为恢复片增加reorient与lower动作。例 yam:20261016_130545_841401d5 从“Position the blue-capped tube over an opening in the orange rack and insert it upright.”改为“Reorient the blue-capped tube held near the orange rack and lower it upright into a rack opening.”。其余29条保留取管插入任务语义。场景49条更新，叙述147→196项，每clip由3项增至4项，点状态均带源帧cutoff。
10条复核总指令语义一致10/10（100%），独立初判4阶段与v2每条4阶段，阶段数差每条0，平均绝对差0。相对v1每条增加1项。此处阶段数是记录的叙述状态数量，未标精确动作分段边界。一致率仅表明本次同会话模型判断一致，同模型家族、独立性较弱、非人工真值。模型精确运行时版本未独立验证，保留requested_model_name与model_name区分。

## 训练建议
49条可候选用于粗粒度训练文本，未发现必须整条剔除的可见证据。全部49条不适合精确洞位、深度、接触时刻或two-tube序列监督。20条恢复片不适合完整松手撤离/成功结果监督，逐clip列表见e92b_training_recommendations.json。标注与复核均只供训练目标或训练池筛选，禁止作测试真值、阈值来源或checkpoint选型依据。
未完成的视觉细节为上述保留的不确定项。本轮要求的抽帧、复判、弱独立复核、统计和文件校验均已完成，无需用户决策。

## 文件与校验
主输出 data/e92_yam_lang_v1/lang_yam_v2.jsonl。v1完整对象逐条保存在v1_value，原文件SHA256保持一致。新增六字段齐全，49唯一ID及parent保持，全部证据帧存在且小于等于cutoff，帧数及容量上限通过。v2 SHA256 9cf9265c6be5ca2664d14cb20c3d8c862c8455e2fee9a0268e83ce2ae869dbdd。
uncertain_resolved采用布尔值，表示整条是否已完全消除不确定；局部解决写入resolved_by_frames与resolution_notes。置信度medium用于粗粒度文本，未校准概率。
抽帧命令使用setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python T/data/e92_yam_lang_v1/e92b_extract.py。只通过H20读取源视频，未连接yam主机。输出位于授权data与本报告路径，公共日志采用锁保护追加。一次浏览在抽帧尚未到达索引28时失败，等待后重读成功。一次JSONL命令参数过长导致SSH失败，确认目标不存在后改用标准输入写入成功，未产生旧结果覆盖。
最终状态 完成。
