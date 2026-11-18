# E99-W2 完整P3 v3.1计划轨迹

已完成固定5000步，使用E98-W A初始化；旧复制入口保留审计，不作为本轮写入入口。

训练：e99w2_driver04.py -> e99w2_train.py；模型/数据/目标：e99w2_model.py、e99w2_data.py、e99w2_objective.py。只用新唯一run；不要重跑已存在的结果路径。

评估：e99w2_eval.py为首轮固定缓存，e99w2_complete_h100.py只补齐原48窗口缺失的47个+100端点，复用E93既有只读特征，不重算旧预测。最终世界指标：results/e99w2_control_20261115_01/evaluation_02_h100_complete；动作/时延：evaluation_01。

GT动作是next-frame反馈，H50与oracle逐元素相等；51..100保持动作49。N1预注册未通过；N8补充结果不得替代N1。全部四时域48窗/6parent；无dev选点或新模型语言。

GPU入口必须在import torch前固定CUDA_VISIBLE_DEVICES，并用e96a.resource.Guard计入E99-W2统一4GPUh账本、校验1.5倍显存余量。完整报告：notes/e99w2_planned_traj_20261115.md。
