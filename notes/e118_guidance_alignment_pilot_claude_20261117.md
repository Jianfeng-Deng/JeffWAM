# E118（Claude 直接执行）任务引导文字 ↔ 图像差分/动作差分 对齐可学习性试点
2026-11-17 BRT。取代 luna 的未执行版（notes/e118_guidance_alignment_pilot_2026-11-17.md）。代码 code_e118/，结果 results/e118_pilot_20261117_02/（data.npz、oof.npz、summary.json、PREREG_LOCK.json），GPU 约 0.05h 量级，新增落盘约 290MB。
## 数据与设计
60 个 train parent，FK 运动文字(horizon 50，arm 分行，eligible)，57,858 样本；文字=SigLIP2 get_text_features(pooler_output) 归一化后冻结；类别：粗类 arm|水平方向|垂直方向|夹爪(154个有效类)、细类加幅度(399)。查询输入：dI=b[a+50]−b[a]（3相机池化2304维）、dA=state[a+25/+50]−state[a]（32维因果本体含位姿，64维）、dI+dA、cur=b[a]、pos=(a/n,a,n)，均拼接 arm one-hot；候选=每类训练折内最常见句子；小MLP→128维+文本线性→128，对称InfoNCE(同类对屏蔽)，类别 1/√频率 采样，2000步，seed 90/91/92 平均；parent 五折 OOF（E102-S折）；指标=类别平衡 top-1/top-5 召回，parent bootstrap 2000(seed90)。判据训练前锁定(PREREG_LOCK.json)。
## 结果（粗类，平衡 top-1 / top-5；随机≈.005/.03）
dI .105/.300；dA .283/.620（平凡）；dI+dA .132/.363；cur .083/.261；pos .030/.118；cur+dI .117/.335；prior .006/.032；shuffled .005/.031。
配对差(top-1)：dI−cur +.021 [.017,.026]；dI−pos +.075；dI−shuffled +.100；dI−prior +.098；cur+dI−cur +.034 [.029,.040]；dA−pos +.253。运动类、细类、top-5 方向一致，均 CI>0。dI+dA 反而低于 dA(−.151)，应是2304维输入淹没64维dA的优化/融合问题，不作结论。
## 判定
按锁定判据：视觉差分对齐“可学习”(dI 对各对照 CI>0，cur+dI>cur)；动作差分可学但属平凡(文字由同一关节位移规则生成)。效应量小：dI 仅比“只看当前”高约2个百分点(平衡top-1)，当前画面本身已携带大部分信息(cur 为随机的约13倍)。
## 局限
只用 train parent 五折，dev 无法评估；文字是规则生成的事后描述，不是希望得到的行为/人写引导；池化特征丢失空间位置，视觉差分可能被低估(未试2×4 token网格差分)；dI+dA 融合未调；阶段→引导规则未检验；类别不平衡强。
## 建议
go（小步）：1) 用未池化 2×4 token 差分重做 dI；2) 检验阶段→引导文字规则（阶段能多大程度预测运动类别）；3) 非动作派生文字(关键帧描述)对 dI 的对齐。
