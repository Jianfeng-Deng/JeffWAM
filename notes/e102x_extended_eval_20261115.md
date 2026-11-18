# E102-X 扩展 dev 评估

状态：COMPLETE；仅推理，三组固定 run03/ckpt_02000.pt。

## 1 合同与窗口预注册
已核对并锁定，见最终评分前锁。

## 2 资源与执行
GPU ≤0.5 GPUh（含失败），新增 ≤10GB；仅复用已有特征。

## 3 指标、分层与配对置信区间
已完成，见末尾最终有效结果。

## 4 结论、未完成和用户决定
已完成，见末尾最终有效结果。

## 窗口锁定（任何模型前向前）
{"brt": "2026-11-15T05:36:12.398234-03:00", "rule": "E93 immutable windows_full.json, all211 candidates; no new anchors; action legality and state checks before any model output; original48 subset", "windows_sha": "074878444310816ee22717601402c0ccae3e34f7010966d95d60333081b417a0", "captures_sha": "a9ffc3b6ec60fdbb54ecc1af81f3ea8b04d07bf14d74471619543fb36e018bb2", "contract_sha": "7ab9e77b5fcdd1252bee92e405acb6ca241725665faa5310b6d523110428a6c5", "rejected": [{"index": 48, "parent": "pc_719305ea3509e357e856", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 49, "parent": "pc_719305ea3509e357e856", "anchor": 136, "reason": "'action'", "type": "KeyError"}, {"index": 50, "parent": "pc_719305ea3509e357e856", "anchor": 252, "reason": "'action'", "type": "KeyError"}, {"index": 51, "parent": "pc_719305ea3509e357e856", "anchor": 364, "reason": "'action'", "type": "KeyError"}, {"index": 52, "parent": "pc_719305ea3509e357e856", "anchor": 480, "reason": "'action'", "type": "KeyError"}, {"index": 53, "parent": "pc_719305ea3509e357e856", "anchor": 592, "reason": "'action'", "type": "KeyError"}, {"index": 54, "parent": "pc_719305ea3509e357e856", "anchor": 708, "reason": "'action'", "type": "KeyError"}, {"index": 55, "parent": "pc_719305ea3509e357e856", "anchor": 824, "reason": "'action'", "type": "KeyError"}, {"index": 56, "parent": "pc_69f4550c2ae996f44dd6", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 57, "parent": "pc_69f4550c2ae996f44dd6", "anchor": 132, "reason": "'action'", "type": "KeyError"}, {"index": 58, "parent": "pc_69f4550c2ae996f44dd6", "anchor": 244, "reason": "'action'", "type": "KeyError"}, {"index": 59, "parent": "pc_69f4550c2ae996f44dd6", "anchor": 352, "reason": "'action'", "type": "KeyError"}, {"index": 60, "parent": "pc_69f4550c2ae996f44dd6", "anchor": 464, "reason": "'action'", "type": "KeyError"}, {"index": 61, "parent": "pc_69f4550c2ae996f44dd6", "anchor": 572, "reason": "'action'", "type": "KeyError"}, {"index": 62, "parent": "pc_69f4550c2ae996f44dd6", "anchor": 684, "reason": "'action'", "type": "KeyError"}, {"index": 63, "parent": "pc_69f4550c2ae996f44dd6", "anchor": 792, "reason": "'action'", "type": "KeyError"}, {"index": 64, "parent": "pc_e177292a707057f5e1b6", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 65, "parent": "pc_e177292a707057f5e1b6", "anchor": 56, "reason": "'action'", "type": "KeyError"}, {"index": 66, "parent": "pc_e177292a707057f5e1b6", "anchor": 88, "reason": "'action'", "type": "KeyError"}, {"index": 67, "parent": "pc_e177292a707057f5e1b6", "anchor": 120, "reason": "'action'", "type": "KeyError"}, {"index": 68, "parent": "pc_e177292a707057f5e1b6", "anchor": 152, "reason": "'action'", "type": "KeyError"}, {"index": 69, "parent": "pc_e177292a707057f5e1b6", "anchor": 188, "reason": "'action'", "type": "KeyError"}, {"index": 70, "parent": "pc_e177292a707057f5e1b6", "anchor": 220, "reason": "'action'", "type": "KeyError"}, {"index": 71, "parent": "pc_e177292a707057f5e1b6", "anchor": 252, "reason": "'action'", "type": "KeyError"}, {"index": 72, "parent": "pc_a2389165922be52eb9c9", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 73, "parent": "pc_a2389165922be52eb9c9", "anchor": 216, "reason": "'action'", "type": "KeyError"}, {"index": 74, "parent": "pc_a2389165922be52eb9c9", "anchor": 408, "reason": "'action'", "type": "KeyError"}, {"index": 75, "parent": "pc_a2389165922be52eb9c9", "anchor": 600, "reason": "'action'", "type": "KeyError"}, {"index": 76, "parent": "pc_a2389165922be52eb9c9", "anchor": 792, "reason": "'action'", "type": "KeyError"}, {"index": 77, "parent": "pc_a2389165922be52eb9c9", "anchor": 984, "reason": "'action'", "type": "KeyError"}, {"index": 78, "parent": "pc_a2389165922be52eb9c9", "anchor": 1176, "reason": "'action'", "type": "KeyError"}, {"index": 79, "parent": "pc_a2389165922be52eb9c9", "anchor": 1368, "reason": "'action'", "type": "KeyError"}, {"index": 80, "parent": "pc_d024b01e3ca473d0ba23", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 81, "parent": "pc_d024b01e3ca473d0ba23", "anchor": 120, "reason": "'action'", "type": "KeyError"}, {"index": 82, "parent": "pc_d024b01e3ca473d0ba23", "anchor": 220, "reason": "'action'", "type": "KeyError"}, {"index": 83, "parent": "pc_d024b01e3ca473d0ba23", "anchor": 320, "reason": "'action'", "type": "KeyError"}, {"index": 84, "parent": "pc_d024b01e3ca473d0ba23", "anchor": 420, "reason": "'action'", "type": "KeyError"}, {"index": 85, "parent": "pc_d024b01e3ca473d0ba23", "anchor": 520, "reason": "'action'", "type": "KeyError"}, {"index": 86, "parent": "pc_d024b01e3ca473d0ba23", "anchor": 620, "reason": "'action'", "type": "KeyError"}, {"index": 87, "parent": "pc_d024b01e3ca473d0ba23", "anchor": 720, "reason": "'action'", "type": "KeyError"}, {"index": 88, "parent": "pc_e69ee076adce2343de83", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 89, "parent": "pc_e69ee076adce2343de83", "anchor": 196, "reason": "'action'", "type": "KeyError"}, {"index": 90, "parent": "pc_e69ee076adce2343de83", "anchor": 372, "reason": "'action'", "type": "KeyError"}, {"index": 91, "parent": "pc_e69ee076adce2343de83", "anchor": 548, "reason": "'action'", "type": "KeyError"}, {"index": 92, "parent": "pc_e69ee076adce2343de83", "anchor": 724, "reason": "'action'", "type": "KeyError"}, {"index": 93, "parent": "pc_e69ee076adce2343de83", "anchor": 900, "reason": "'action'", "type": "KeyError"}, {"index": 94, "parent": "pc_e69ee076adce2343de83", "anchor": 1076, "reason": "'action'", "type": "KeyError"}, {"index": 95, "parent": "pc_e69ee076adce2343de83", "anchor": 1252, "reason": "'action'", "type": "KeyError"}, {"index": 96, "parent": "pc_2ac3ef86daa61d42a3f0", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 97, "parent": "pc_2ac3ef86daa61d42a3f0", "anchor": 132, "reason": "'action'", "type": "KeyError"}, {"index": 98, "parent": "pc_2ac3ef86daa61d42a3f0", "anchor": 244, "reason": "'action'", "type": "KeyError"}, {"index": 99, "parent": "pc_2ac3ef86daa61d42a3f0", "anchor": 356, "reason": "'action'", "type": "KeyError"}, {"index": 100, "parent": "pc_2ac3ef86daa61d42a3f0", "anchor": 464, "reason": "'action'", "type": "KeyError"}, {"index": 101, "parent": "pc_2ac3ef86daa61d42a3f0", "anchor": 576, "reason": "'action'", "type": "KeyError"}, {"index": 102, "parent": "pc_2ac3ef86daa61d42a3f0", "anchor": 688, "reason": "'action'", "type": "KeyError"}, {"index": 103, "parent": "pc_2ac3ef86daa61d42a3f0", "anchor": 796, "reason": "'action'", "type": "KeyError"}, {"index": 104, "parent": "pc_896e79237a36a8758aa2", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 105, "parent": "pc_896e79237a36a8758aa2", "anchor": 288, "reason": "'action'", "type": "KeyError"}, {"index": 106, "parent": "pc_896e79237a36a8758aa2", "anchor": 556, "reason": "'action'", "type": "KeyError"}, {"index": 107, "parent": "pc_896e79237a36a8758aa2", "anchor": 820, "reason": "'action'", "type": "KeyError"}, {"index": 108, "parent": "pc_896e79237a36a8758aa2", "anchor": 1088, "reason": "'action'", "type": "KeyError"}, {"index": 109, "parent": "pc_896e79237a36a8758aa2", "anchor": 1356, "reason": "'action'", "type": "KeyError"}, {"index": 110, "parent": "pc_896e79237a36a8758aa2", "anchor": 1620, "reason": "'action'", "type": "KeyError"}, {"index": 111, "parent": "pc_896e79237a36a8758aa2", "anchor": 1888, "reason": "'action'", "type": "KeyError"}, {"index": 112, "parent": "pc_e7fe9b7576da778a819e", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 113, "parent": "pc_e7fe9b7576da778a819e", "anchor": 224, "reason": "'action'", "type": "KeyError"}, {"index": 114, "parent": "pc_e7fe9b7576da778a819e", "anchor": 424, "reason": "'action'", "type": "KeyError"}, {"index": 115, "parent": "pc_e7fe9b7576da778a819e", "anchor": 624, "reason": "'action'", "type": "KeyError"}, {"index": 116, "parent": "pc_e7fe9b7576da778a819e", "anchor": 824, "reason": "'action'", "type": "KeyError"}, {"index": 117, "parent": "pc_e7fe9b7576da778a819e", "anchor": 1024, "reason": "'action'", "type": "KeyError"}, {"index": 118, "parent": "pc_e7fe9b7576da778a819e", "anchor": 1224, "reason": "'action'", "type": "KeyError"}, {"index": 119, "parent": "pc_e7fe9b7576da778a819e", "anchor": 1424, "reason": "'action'", "type": "KeyError"}, {"index": 120, "parent": "pc_989e0194750029688590", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 121, "parent": "pc_989e0194750029688590", "anchor": 132, "reason": "'action'", "type": "KeyError"}, {"index": 122, "parent": "pc_989e0194750029688590", "anchor": 240, "reason": "'action'", "type": "KeyError"}, {"index": 123, "parent": "pc_989e0194750029688590", "anchor": 348, "reason": "'action'", "type": "KeyError"}, {"index": 124, "parent": "pc_989e0194750029688590", "anchor": 460, "reason": "'action'", "type": "KeyError"}, {"index": 125, "parent": "pc_989e0194750029688590", "anchor": 568, "reason": "'action'", "type": "KeyError"}, {"index": 126, "parent": "pc_989e0194750029688590", "anchor": 676, "reason": "'action'", "type": "KeyError"}, {"index": 127, "parent": "pc_989e0194750029688590", "anchor": 788, "reason": "'action'", "type": "KeyError"}, {"index": 128, "parent": "pc_93163756047dd9e1b779", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 129, "parent": "pc_93163756047dd9e1b779", "anchor": 36, "reason": "'action'", "type": "KeyError"}, {"index": 130, "parent": "pc_93163756047dd9e1b779", "anchor": 48, "reason": "'action'", "type": "KeyError"}, {"index": 131, "parent": "pc_93163756047dd9e1b779", "anchor": 60, "reason": "'action'", "type": "KeyError"}, {"index": 132, "parent": "pc_93163756047dd9e1b779", "anchor": 72, "reason": "'action'", "type": "KeyError"}, {"index": 133, "parent": "pc_93163756047dd9e1b779", "anchor": 84, "reason": "'action'", "type": "KeyError"}, {"index": 134, "parent": "pc_93163756047dd9e1b779", "anchor": 100, "reason": "'action'", "type": "KeyError"}, {"index": 135, "parent": "pc_93163756047dd9e1b779", "anchor": 112, "reason": "'action'", "type": "KeyError"}, {"index": 136, "parent": "pc_d645d827e563a579e5b5", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 137, "parent": "pc_d645d827e563a579e5b5", "anchor": 28, "reason": "'action'", "type": "KeyError"}, {"index": 138, "parent": "pc_d645d827e563a579e5b5", "anchor": 32, "reason": "'action'", "type": "KeyError"}, {"index": 139, "parent": "pc_dee4ab5bde2af06bde81", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 140, "parent": "pc_dee4ab5bde2af06bde81", "anchor": 52, "reason": "'action'", "type": "KeyError"}, {"index": 141, "parent": "pc_dee4ab5bde2af06bde81", "anchor": 84, "reason": "'action'", "type": "KeyError"}, {"index": 142, "parent": "pc_dee4ab5bde2af06bde81", "anchor": 116, "reason": "'action'", "type": "KeyError"}, {"index": 143, "parent": "pc_dee4ab5bde2af06bde81", "anchor": 148, "reason": "'action'", "type": "KeyError"}, {"index": 144, "parent": "pc_dee4ab5bde2af06bde81", "anchor": 176, "reason": "'action'", "type": "KeyError"}, {"index": 145, "parent": "pc_dee4ab5bde2af06bde81", "anchor": 208, "reason": "'action'", "type": "KeyError"}, {"index": 146, "parent": "pc_dee4ab5bde2af06bde81", "anchor": 240, "reason": "'action'", "type": "KeyError"}, {"index": 147, "parent": "pc_7e7b988a55b73bfbb71b", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 148, "parent": "pc_7e7b988a55b73bfbb71b", "anchor": 140, "reason": "'action'", "type": "KeyError"}, {"index": 149, "parent": "pc_7e7b988a55b73bfbb71b", "anchor": 256, "reason": "'action'", "type": "KeyError"}, {"index": 150, "parent": "pc_7e7b988a55b73bfbb71b", "anchor": 372, "reason": "'action'", "type": "KeyError"}, {"index": 151, "parent": "pc_7e7b988a55b73bfbb71b", "anchor": 488, "reason": "'action'", "type": "KeyError"}, {"index": 152, "parent": "pc_7e7b988a55b73bfbb71b", "anchor": 604, "reason": "'action'", "type": "KeyError"}, {"index": 153, "parent": "pc_7e7b988a55b73bfbb71b", "anchor": 720, "reason": "'action'", "type": "KeyError"}, {"index": 154, "parent": "pc_7e7b988a55b73bfbb71b", "anchor": 836, "reason": "'action'", "type": "KeyError"}, {"index": 155, "parent": "pc_67b50a5790504d01056e", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 156, "parent": "pc_67b50a5790504d01056e", "anchor": 32, "reason": "'action'", "type": "KeyError"}, {"index": 157, "parent": "pc_67b50a5790504d01056e", "anchor": 44, "reason": "'action'", "type": "KeyError"}, {"index": 158, "parent": "pc_67b50a5790504d01056e", "anchor": 52, "reason": "'action'", "type": "KeyError"}, {"index": 159, "parent": "pc_67b50a5790504d01056e", "anchor": 64, "reason": "'action'", "type": "KeyError"}, {"index": 160, "parent": "pc_67b50a5790504d01056e", "anchor": 76, "reason": "'action'", "type": "KeyError"}, {"index": 161, "parent": "pc_67b50a5790504d01056e", "anchor": 84, "reason": "'action'", "type": "KeyError"}, {"index": 162, "parent": "pc_67b50a5790504d01056e", "anchor": 96, "reason": "'action'", "type": "KeyError"}, {"index": 163, "parent": "pc_c830ce752de00cadf7b5", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 164, "parent": "pc_c830ce752de00cadf7b5", "anchor": 368, "reason": "'action'", "type": "KeyError"}, {"index": 165, "parent": "pc_c830ce752de00cadf7b5", "anchor": 716, "reason": "'action'", "type": "KeyError"}, {"index": 166, "parent": "pc_c830ce752de00cadf7b5", "anchor": 1064, "reason": "'action'", "type": "KeyError"}, {"index": 167, "parent": "pc_c830ce752de00cadf7b5", "anchor": 1408, "reason": "'action'", "type": "KeyError"}, {"index": 168, "parent": "pc_c830ce752de00cadf7b5", "anchor": 1756, "reason": "'action'", "type": "KeyError"}, {"index": 169, "parent": "pc_c830ce752de00cadf7b5", "anchor": 2104, "reason": "'action'", "type": "KeyError"}, {"index": 170, "parent": "pc_c830ce752de00cadf7b5", "anchor": 2448, "reason": "'action'", "type": "KeyError"}, {"index": 171, "parent": "pc_1998c0f1eceb0a6a3286", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 172, "parent": "pc_1998c0f1eceb0a6a3286", "anchor": 148, "reason": "'action'", "type": "KeyError"}, {"index": 173, "parent": "pc_1998c0f1eceb0a6a3286", "anchor": 276, "reason": "'action'", "type": "KeyError"}, {"index": 174, "parent": "pc_1998c0f1eceb0a6a3286", "anchor": 400, "reason": "'action'", "type": "KeyError"}, {"index": 175, "parent": "pc_1998c0f1eceb0a6a3286", "anchor": 528, "reason": "'action'", "type": "KeyError"}, {"index": 176, "parent": "pc_1998c0f1eceb0a6a3286", "anchor": 656, "reason": "'action'", "type": "KeyError"}, {"index": 177, "parent": "pc_1998c0f1eceb0a6a3286", "anchor": 780, "reason": "'action'", "type": "KeyError"}, {"index": 178, "parent": "pc_1998c0f1eceb0a6a3286", "anchor": 908, "reason": "'action'", "type": "KeyError"}, {"index": 179, "parent": "pc_4c5129844edcb591cdf8", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 180, "parent": "pc_4c5129844edcb591cdf8", "anchor": 424, "reason": "'action'", "type": "KeyError"}, {"index": 181, "parent": "pc_4c5129844edcb591cdf8", "anchor": 624, "reason": "'action'", "type": "KeyError"}, {"index": 182, "parent": "pc_4c5129844edcb591cdf8", "anchor": 828, "reason": "'action'", "type": "KeyError"}, {"index": 183, "parent": "pc_4c5129844edcb591cdf8", "anchor": 1028, "reason": "'action'", "type": "KeyError"}, {"index": 184, "parent": "pc_4c5129844edcb591cdf8", "anchor": 1228, "reason": "'action'", "type": "KeyError"}, {"index": 185, "parent": "pc_4c5129844edcb591cdf8", "anchor": 1432, "reason": "'action'", "type": "KeyError"}, {"index": 186, "parent": "pc_4c5129844edcb591cdf8", "anchor": 1632, "reason": "'action'", "type": "KeyError"}, {"index": 187, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 188, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 336, "reason": "'action'", "type": "KeyError"}, {"index": 189, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 652, "reason": "'action'", "type": "KeyError"}, {"index": 190, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 964, "reason": "'action'", "type": "KeyError"}, {"index": 191, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 1280, "reason": "'action'", "type": "KeyError"}, {"index": 192, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 1592, "reason": "'action'", "type": "KeyError"}, {"index": 193, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 1908, "reason": "'action'", "type": "KeyError"}, {"index": 194, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 2220, "reason": "'action'", "type": "KeyError"}, {"index": 195, "parent": "pc_2b6feb101bc28426c698", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 196, "parent": "pc_2b6feb101bc28426c698", "anchor": 348, "reason": "'action'", "type": "KeyError"}, {"index": 197, "parent": "pc_2b6feb101bc28426c698", "anchor": 672, "reason": "'action'", "type": "KeyError"}, {"index": 198, "parent": "pc_2b6feb101bc28426c698", "anchor": 996, "reason": "'action'", "type": "KeyError"}, {"index": 199, "parent": "pc_2b6feb101bc28426c698", "anchor": 1320, "reason": "'action'", "type": "KeyError"}, {"index": 200, "parent": "pc_2b6feb101bc28426c698", "anchor": 1648, "reason": "'action'", "type": "KeyError"}, {"index": 201, "parent": "pc_2b6feb101bc28426c698", "anchor": 1972, "reason": "'action'", "type": "KeyError"}, {"index": 202, "parent": "pc_2b6feb101bc28426c698", "anchor": 2296, "reason": "'action'", "type": "KeyError"}, {"index": 203, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 24, "reason": "'action'", "type": "KeyError"}, {"index": 204, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 132, "reason": "'action'", "type": "KeyError"}, {"index": 205, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 244, "reason": "'action'", "type": "KeyError"}, {"index": 206, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 352, "reason": "'action'", "type": "KeyError"}, {"index": 207, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 464, "reason": "'action'", "type": "KeyError"}, {"index": 208, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 576, "reason": "'action'", "type": "KeyError"}, {"index": 209, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 684, "reason": "'action'", "type": "KeyError"}, {"index": 210, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 796, "reason": "'action'", "type": "KeyError"}], "bootstrap": {"unit": "parent_equal", "replicates": 2000, "seed": 90}, "horizons": [10, 25, 50, 100], "flow_steps": 4, "inference_seed": 90, "language": "short human default only", "peak_estimate_mib": 6144, "gpu_budget_s": 1800, "disk_limit_bytes": 10737418240, "feature_extraction_bytes": 0, "decision": "E102-A unchanged joint both L1 improvement CI lower>0; gripper both A1/A2 F1@10 minus A0 CI lower>0 and lower mean false switches; world N1-no CI lower>0 at25/50/100"}

|parent|候选|动作合法|+10/+25/+50/+100|
|---|---:|---:|---|
|pc_013cc6b0e0118d072947|8|8|8/8/8/8|
|pc_1998c0f1eceb0a6a3286|8|0|0/0/0/0|
|pc_29c4de823981ef82b501|8|8|8/8/8/8|
|pc_2ac3ef86daa61d42a3f0|8|0|0/0/0/0|
|pc_2b6feb101bc28426c698|8|0|0/0/0/0|
|pc_3b0a951de0881e96822f|8|8|8/8/8/8|
|pc_4c5129844edcb591cdf8|8|0|0/0/0/0|
|pc_5763582dd74c2c8cdfad|8|0|0/0/0/0|
|pc_60067d4b6b3143f5041d|8|8|8/8/8/8|
|pc_67b50a5790504d01056e|8|0|0/0/0/0|
|pc_69f4550c2ae996f44dd6|8|0|0/0/0/0|
|pc_719305ea3509e357e856|8|0|0/0/0/0|
|pc_7e7b988a55b73bfbb71b|8|0|0/0/0/0|
|pc_896e79237a36a8758aa2|8|0|0/0/0/0|
|pc_93163756047dd9e1b779|8|0|0/0/0/0|
|pc_989e0194750029688590|8|0|0/0/0/0|
|pc_a2389165922be52eb9c9|8|0|0/0/0/0|
|pc_ba45eef0f6b112ab71df|8|0|0/0/0/0|
|pc_c830ce752de00cadf7b5|8|0|0/0/0/0|
|pc_c9c7158eb5f7b23c73ca|8|8|8/8/8/8|
|pc_d024b01e3ca473d0ba23|8|0|0/0/0/0|
|pc_d645d827e563a579e5b5|3|0|0/0/0/0|
|pc_dee4ab5bde2af06bde81|8|0|0/0/0/0|
|pc_e177292a707057f5e1b6|8|0|0/0/0/0|
|pc_e69ee076adce2343de83|8|0|0/0/0/0|
|pc_e7fe9b7576da778a819e|8|0|0/0/0/0|
|pc_fb991bc8666952f1f50f|8|8|8/8/8/8|

原48窗完整包含；所有评估条件沿用E102-A，包括训练拟合夹爪合同、起点代理、一一事件匹配。状态/动作失败记录见锁文件；世界时域按有效mask，缺+100不伪造。单seed，仅dev结论，无部署操作。

## 窗口锁定（任何模型前向前）
{"brt": "2026-11-15T05:38:07.287151-03:00", "rule": "E93 immutable windows_full.json, all211 candidates; no new anchors; action legality and state checks before any model output; original48 subset", "windows_sha": "074878444310816ee22717601402c0ccae3e34f7010966d95d60333081b417a0", "captures_sha": "a9ffc3b6ec60fdbb54ecc1af81f3ea8b04d07bf14d74471619543fb36e018bb2", "contract_sha": "7ab9e77b5fcdd1252bee92e405acb6ca241725665faa5310b6d523110428a6c5", "rejected": [{"index": 163, "parent": "pc_c830ce752de00cadf7b5", "anchor": 24, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 164, "parent": "pc_c830ce752de00cadf7b5", "anchor": 368, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 165, "parent": "pc_c830ce752de00cadf7b5", "anchor": 716, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 166, "parent": "pc_c830ce752de00cadf7b5", "anchor": 1064, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 167, "parent": "pc_c830ce752de00cadf7b5", "anchor": 1408, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 168, "parent": "pc_c830ce752de00cadf7b5", "anchor": 1756, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 169, "parent": "pc_c830ce752de00cadf7b5", "anchor": 2104, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 170, "parent": "pc_c830ce752de00cadf7b5", "anchor": 2448, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 187, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 24, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 188, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 336, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 189, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 652, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 190, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 964, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 191, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 1280, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 192, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 1592, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 193, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 1908, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 194, "parent": "pc_ba45eef0f6b112ab71df", "anchor": 2220, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 195, "parent": "pc_2b6feb101bc28426c698", "anchor": 24, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 196, "parent": "pc_2b6feb101bc28426c698", "anchor": 348, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 197, "parent": "pc_2b6feb101bc28426c698", "anchor": 672, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 198, "parent": "pc_2b6feb101bc28426c698", "anchor": 996, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 199, "parent": "pc_2b6feb101bc28426c698", "anchor": 1320, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 200, "parent": "pc_2b6feb101bc28426c698", "anchor": 1648, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 201, "parent": "pc_2b6feb101bc28426c698", "anchor": 1972, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 202, "parent": "pc_2b6feb101bc28426c698", "anchor": 2296, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 203, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 24, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 204, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 132, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 205, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 244, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 206, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 352, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 207, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 464, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 208, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 576, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 209, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 684, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}, {"index": 210, "parent": "pc_5763582dd74c2c8cdfad", "anchor": 796, "reason": "Unresolved demo6 stratum; no cached image metadata", "type": "ValueError"}], "bootstrap": {"unit": "parent_equal", "replicates": 2000, "seed": 90}, "horizons": [10, 25, 50, 100], "flow_steps": 4, "inference_seed": 90, "language": "short human default only", "peak_estimate_mib": 6144, "gpu_budget_s": 1800, "disk_limit_bytes": 10737418240, "feature_extraction_bytes": 0, "decision": "E102-A unchanged joint both L1 improvement CI lower>0; gripper both A1/A2 F1@10 minus A0 CI lower>0 and lower mean false switches; world N1-no CI lower>0 at25/50/100"}

|parent|候选|动作合法|+10/+25/+50/+100|
|---|---:|---:|---|
|pc_013cc6b0e0118d072947|8|8|8/8/8/8|
|pc_1998c0f1eceb0a6a3286|8|8|8/8/8/8|
|pc_29c4de823981ef82b501|8|8|8/8/8/8|
|pc_2ac3ef86daa61d42a3f0|8|8|8/8/8/8|
|pc_2b6feb101bc28426c698|8|0|8/8/8/8|
|pc_3b0a951de0881e96822f|8|8|8/8/8/8|
|pc_4c5129844edcb591cdf8|8|8|8/8/8/8|
|pc_5763582dd74c2c8cdfad|8|0|8/8/8/8|
|pc_60067d4b6b3143f5041d|8|8|8/8/8/8|
|pc_67b50a5790504d01056e|8|8|8/8/8/8|
|pc_69f4550c2ae996f44dd6|8|8|8/8/8/8|
|pc_719305ea3509e357e856|8|8|8/8/8/8|
|pc_7e7b988a55b73bfbb71b|8|8|8/8/8/8|
|pc_896e79237a36a8758aa2|8|8|8/8/8/8|
|pc_93163756047dd9e1b779|8|8|8/8/8/8|
|pc_989e0194750029688590|8|8|8/8/8/8|
|pc_a2389165922be52eb9c9|8|8|8/8/8/8|
|pc_ba45eef0f6b112ab71df|8|0|8/8/8/8|
|pc_c830ce752de00cadf7b5|8|0|8/8/8/8|
|pc_c9c7158eb5f7b23c73ca|8|8|8/8/8/8|
|pc_d024b01e3ca473d0ba23|8|8|8/8/8/8|
|pc_d645d827e563a579e5b5|3|3|3/3/3/0|
|pc_dee4ab5bde2af06bde81|8|8|8/8/8/8|
|pc_e177292a707057f5e1b6|8|8|8/8/8/8|
|pc_e69ee076adce2343de83|8|8|8/8/8/8|
|pc_e7fe9b7576da778a819e|8|8|8/8/8/8|
|pc_fb991bc8666952f1f50f|8|8|8/8/8/8|

原48窗完整包含；所有评估条件沿用E102-A，包括训练拟合夹爪合同、起点代理、一一事件匹配。状态/动作失败记录见锁文件；世界时域按有效mask，缺+100不伪造。单seed，仅dev结论，无部署操作。

## 窗口锁定（任何模型前向前）
{"brt": "2026-11-15T05:38:34.310026-03:00", "rule": "E93 immutable windows_full.json, all211 candidates; no new anchors; action legality and state checks before any model output; original48 subset", "windows_sha": "074878444310816ee22717601402c0ccae3e34f7010966d95d60333081b417a0", "captures_sha": "a9ffc3b6ec60fdbb54ecc1af81f3ea8b04d07bf14d74471619543fb36e018bb2", "contract_sha": "7ab9e77b5fcdd1252bee92e405acb6ca241725665faa5310b6d523110428a6c5", "rejected": [], "bootstrap": {"unit": "parent_equal", "replicates": 2000, "seed": 90}, "horizons": [10, 25, 50, 100], "flow_steps": 4, "inference_seed": 90, "language": "short human default only", "peak_estimate_mib": 6144, "gpu_budget_s": 1800, "disk_limit_bytes": 10737418240, "feature_extraction_bytes": 0, "decision": "E102-A unchanged joint both L1 improvement CI lower>0; gripper both A1/A2 F1@10 minus A0 CI lower>0 and lower mean false switches; world N1-no CI lower>0 at25/50/100"}

|parent|候选|动作合法|+10/+25/+50/+100|
|---|---:|---:|---|
|pc_013cc6b0e0118d072947|8|8|8/8/8/8|
|pc_1998c0f1eceb0a6a3286|8|8|8/8/8/8|
|pc_29c4de823981ef82b501|8|8|8/8/8/8|
|pc_2ac3ef86daa61d42a3f0|8|8|8/8/8/8|
|pc_2b6feb101bc28426c698|8|8|8/8/8/8|
|pc_3b0a951de0881e96822f|8|8|8/8/8/8|
|pc_4c5129844edcb591cdf8|8|8|8/8/8/8|
|pc_5763582dd74c2c8cdfad|8|8|8/8/8/8|
|pc_60067d4b6b3143f5041d|8|8|8/8/8/8|
|pc_67b50a5790504d01056e|8|8|8/8/8/8|
|pc_69f4550c2ae996f44dd6|8|8|8/8/8/8|
|pc_719305ea3509e357e856|8|8|8/8/8/8|
|pc_7e7b988a55b73bfbb71b|8|8|8/8/8/8|
|pc_896e79237a36a8758aa2|8|8|8/8/8/8|
|pc_93163756047dd9e1b779|8|8|8/8/8/8|
|pc_989e0194750029688590|8|8|8/8/8/8|
|pc_a2389165922be52eb9c9|8|8|8/8/8/8|
|pc_ba45eef0f6b112ab71df|8|8|8/8/8/8|
|pc_c830ce752de00cadf7b5|8|8|8/8/8/8|
|pc_c9c7158eb5f7b23c73ca|8|8|8/8/8/8|
|pc_d024b01e3ca473d0ba23|8|8|8/8/8/8|
|pc_d645d827e563a579e5b5|3|3|3/3/3/0|
|pc_dee4ab5bde2af06bde81|8|8|8/8/8/8|
|pc_e177292a707057f5e1b6|8|8|8/8/8/8|
|pc_e69ee076adce2343de83|8|8|8/8/8/8|
|pc_e7fe9b7576da778a819e|8|8|8/8/8/8|
|pc_fb991bc8666952f1f50f|8|8|8/8/8/8|

原48窗完整包含；所有评估条件沿用E102-A，包括训练拟合夹爪合同、起点代理、一一事件匹配。状态/动作失败记录见锁文件；世界时域按有效mask，缺+100不伪造。单seed，仅dev结论，无部署操作。

## 最终评分前锁与预检更正
前两次仅CPU预检，无CUDA或模型输出。PREFLIGHT01发现E93省略action数组；通过只读E97 dev_state_restore_receipt与SHA核验27capture的原始反馈，将缺失action仅在RAM按原original_action_index=i+1引用q[1:]，已有动作逐元素验证不变。无标签移动、插值或预测补全。PREFLIGHT02中旧stratum工具缺元数据抛错，修正为source_collection的demo3/4/6，不作为合法性剔除。最终锁覆盖211窗/27parent、原48窗完整子集；所有本体/动作检查通过。+100按合法mask，所有窗口其他三时域齐全。新增特征0字节。最终窗口表以上面第三次锁为准，前两次保留为审计。

## 资源监控并发写入失败与恢复（计入预算）
首段A0正常完成。A1首段的主线程与监控线程同时通过旧atomic函数写同一个PID临时文件，触发FileNotFoundError，已在finally关闭资源段并退出；非OOM、非数据或模型选择。失败原目录A1、日志、worker_failed01.py、SCHEDULER_failed01.json完整保留。仅对E102-X Guard.check状态写入加RLock；在新A1_retry02/A2_retry02目录运行，A0不重跑。四段全过程墙钟各加30s余量计入0.5GPUh预算。无模型、窗口、终点或指标改变。

## 扩展评估最终结果（以下为最终有效口径）
完成：2026-11-15T05:43:03.284658-03:00 BRT。三组固定2000终点，无训练；parent等权bootstrap2000 seed90。211窗/27parent，+10/+25/+50各211窗/27parent，+100为208窗/26parent（demo4短episode 3窗缺+100）；不填补缺失。原48窗/6parent完整子集，单独结果见SUMMARY.json original48。
事件真值为反馈运动起点代理，沿用训练拟合阈值/起点规则。事件按预测窗口暴露计数，重叠可能重复；无事件parent的P/R/F1按原合同为0。gt_action为下一帧反馈关节+起点两态夹爪；oracle为未来实测阈值两态，二者夹爪时机不同。所有夹爪固定开度沿用训练合同。

### 全27parent关节与夹爪
|组|前8 L1/复制 [改善CI]|H50 L1/复制 [改善CI]|前8/H50 MSE skill %|状态准确率/保持|F1 ±5/±10|真事件窗口数/物理去重|
|---|---|---|---|---|---|---|
|A0|0.011445/0.010717 [-0.0010106196366283934, -0.000447316486652992]|0.058769/0.059653 [-0.0009298648062586504, 0.0030907064025342355]|8.187/16.113|0.8836/0.9297|0.0787/0.1201|49/48|
|A1|0.011899/0.010717 [-0.0016962319819964688, -0.0007167076891108241]|0.062693/0.059653 [-0.005671309061519547, -0.000245820394067705]|14.819/18.911|0.9481/0.9297|0.1290/0.2704|49/48|
|A2|0.009193/0.010717 [0.0006112022502607652, 0.0024386647323818445]|0.046324/0.059653 [0.008385734253742077, 0.01850229968582457]|58.090/58.651|0.9402/0.9297|0.1785/0.2612|49/48|

|组|容差|P/R/F1|F1 CI|假切换/分钟 [CI]|时误差abs中位/P90、signed中位|匹配数/预测数|
|---|---|---|---|---|---|---|
|A0|±5|0.0649/0.1204/0.0787|[0.028571428571428567, 0.14247768692213136]|19.667 [13.499999999999998, 26.170833333333302]|2.5/4.3、-2.5|8/126|
|A0|±10|0.1020/0.1698/0.1201|[0.055313852813852815, 0.19644821228154558]|19.000 [12.83333333333333, 25.666666666666657]|3.5/7.9、-3.0|12/126|
|A1|±5|0.1420/0.1327/0.1290|[0.043209876543209874, 0.22530864197530864]|6.833 [4.995833333333332, 8.999999999999998]|2.0/4.3、-0.5|8/49|
|A1|±10|0.3056/0.2654/0.2704|[0.15123456790123455, 0.3926234567901233]|5.333 [3.499999999999999, 7.333333333333331]|6.0/9.0、0.0|17/49|
|A2|±5|0.1667/0.2068/0.1785|[0.07848324514991181, 0.2975308641975309]|10.222 [7.499999999999998, 13.333333333333329]|3.0/5.0、-1.0|11/69|
|A2|±10|0.2469/0.2932/0.2612|[0.13580246913580246, 0.40195326278659604]|9.389 [6.499999999999999, 12.72222222222222]|4.0/7.0、0.5|16/69|

### 世界头（skill及配对差用百分点pp；保留率%）
|组|h|GT / pred N1 / no-traj / oracle %|N1−no [CI] pp|保留率 %|
|---|---:|---|---|---|
|A0|10|14.433/10.062/4.437/14.442|5.625 [4.543,6.628]|56.22|
|A0|25|23.556/16.282/6.768/23.536|9.514 [7.780,11.346]|56.74|
|A0|50|30.962/20.332/9.314/30.979|11.017 [9.079,13.173]|50.85|
|A0|100|43.921/33.867/17.500/45.750|16.367 [13.662,19.350]|57.94|
|A1|10|13.040/10.299/5.372/13.039|4.927 [3.754,6.038]|64.27|
|A1|25|21.517/17.198/8.711/21.504|8.487 [6.639,10.415]|66.34|
|A1|50|28.370/19.867/11.819/28.391|8.048 [5.955,10.027]|48.56|
|A1|100|42.741/34.777/20.401/44.014|14.376 [11.666,17.032]|60.88|
|A2|10|12.420/10.802/4.279/12.432|6.523 [5.114,7.794]|80.00|
|A2|25|21.067/18.756/6.550/21.055|12.206 [10.419,13.926]|84.15|
|A2|50|28.313/23.293/8.880/28.313|14.413 [12.171,16.455]|74.17|
|A2|100|41.637/37.510/16.198/43.619|21.312 [18.011,24.673]|77.72|

### 重点(a)：夹爪配对差
|组−A0|容差|F1差 [CI]|假切换差/分钟 [CI]|预注册有效|
|---|---|---|---|---|
|A1−A0|±5|0.05029 [-0.04933174335952113, 0.15438953022286347]|-12.833 [-19.999999999999996, -6.333333333333333]|False|
|A1−A0|±10|0.15029 [0.028327387632943193, 0.27422505478061027]|-13.667 [-20.999999999999996, -6.999999999999998]|True|
|A2−A0|±5|0.09976 [-0.007533068783068784, 0.21435619421730523]|-9.444 [-14.999999999999996, -3.9444444444444438]|False|
|A2−A0|±10|0.14112 [0.007904441237774583, 0.2776028806584362]|-9.611 [-15.277777777777773, -3.777777777777777]|True|

### 重点(b)：A2−A0 gt_action 世界skill配对差
|h|差 pp|95%CI pp|parent|
|---|---:|---|---:|
|10|-2.012|[-2.251,-1.739]|27|
|25|-2.489|[-2.769,-2.199]|27|
|50|-2.649|[-2.936,-2.352]|27|
|100|-2.284|[-2.934,-1.667]|26|

### 重点(c)：沿用E102-A预注册机械判定
{"joint_by_group": {"A0": false, "A1": false, "A2": true}, "world_by_group": {"A0": true, "A1": true, "A2": true}, "gripper_by_group": {"A1": true, "A2": true}, "gripper_fix_effective": true}

### 按demo3/4/6分层附表（不用于增设判据）
|层|组|parent/窗|前8/H50 L1|前8/H50 MSE skill %|状态准确率|F1±5/±10|P/R±10|假切换/分钟±10|事件窗口/去重|
|---|---|---|---|---|---|---|---|---|
|demo3|A0|8/64|0.009334/0.049016|14.464/11.044|0.9244|0.0477/0.0894|0.0608/0.1875|24.187|13/12|
|demo3|A1|8/64|0.009700/0.052216|23.480/16.337|0.9614|0.2188/0.2812|0.3333/0.2812|6.187|13/12|
|demo3|A2|8/64|0.008286/0.040140|60.556/58.658|0.9527|0.1125/0.2250|0.2083/0.2500|7.312|13/12|
|demo4|A0|12/91|0.012204/0.061159|5.268/19.893|0.8702|0.0810/0.1019|0.1000/0.1042|18.375|18/18|
|demo4|A1|12/91|0.012960/0.066432|6.798/18.928|0.9366|0.0000/0.1528|0.2083/0.1250|3.750|18/18|
|demo4|A2|12/91|0.010074/0.050800|51.440/50.620|0.9184|0.1071/0.1905|0.1806/0.2292|12.125|18/18|
|demo6|A0|7/56|0.012557/0.065818|6.193/15.769|0.8600|0.1102/0.1864|0.1524/0.2619|14.143|18/18|
|demo6|A1|7/56|0.012592/0.068257|18.592/21.343|0.9529|0.2476/0.4595|0.4405/0.4881|7.071|18/18|
|demo6|A2|7/56|0.008721/0.045718|66.297/69.652|0.9632|0.3762/0.4238|0.4048/0.4524|7.071|18/18|

|层|组|h|GT / pred N1 / no / oracle skill %|N1−no pp [CI]|保留率%|
|---|---|---|---|---|---|
|demo3|A0|10|14.779/9.154/4.045/14.665|5.108 [3.662,6.617]|48.10|
|demo3|A0|25|22.202/14.677/6.077/22.198|8.600 [7.076,10.033]|53.35|
|demo3|A0|50|29.610/19.885/8.394/29.566|11.491 [7.939,14.896]|54.28|
|demo3|A0|100|41.754/30.780/14.903/44.040|15.876 [7.720,23.566]|54.49|
|demo3|A1|10|13.657/9.707/5.085/13.534|4.623 [2.877,6.610]|54.71|
|demo3|A1|25|20.749/16.060/7.915/20.754|8.145 [6.458,9.432]|63.44|
|demo3|A1|50|27.584/19.836/11.142/27.556|8.694 [4.178,12.491]|52.97|
|demo3|A1|100|41.288/33.312/19.429/42.645|13.883 [5.972,20.474]|59.80|
|demo3|A2|10|12.583/10.708/4.197/12.513|6.511 [4.652,8.253]|78.29|
|demo3|A2|25|20.072/18.470/6.485/20.074|11.984 [9.392,14.484]|88.20|
|demo3|A2|50|27.398/23.584/8.275/27.364|15.309 [11.106,19.313]|80.20|
|demo3|A2|100|39.463/35.993/13.640/41.964|22.353 [12.926,29.975]|78.92|
|demo4|A0|10|13.916/9.592/4.838/13.932|4.754 [2.918,6.533]|52.28|
|demo4|A0|25|24.471/16.697/7.545/24.420|9.153 [5.664,12.961]|54.24|
|demo4|A0|50|30.453/19.794/10.093/30.492|9.701 [5.885,14.395]|47.56|
|demo4|A0|100|43.007/32.829/15.702/44.817|17.127 [12.955,20.616]|58.83|
|demo4|A1|10|12.194/9.662/6.178/12.199|3.484 [1.885,5.107]|57.87|
|demo4|A1|25|21.964/17.220/10.388/21.934|6.832 [3.624,10.554]|59.17|
|demo4|A1|50|27.584/18.761/12.744/27.629|6.017 [3.006,9.532]|40.42|
|demo4|A1|100|41.768/33.514/19.393/43.021|14.121 [10.174,17.220]|59.76|
|demo4|A2|10|12.000/9.833/5.094/12.017|4.739 [2.737,6.332]|68.45|
|demo4|A2|25|22.018/19.027/7.430/21.992|11.597 [8.388,14.767]|79.64|
|demo4|A2|50|28.095/21.573/9.152/28.107|12.421 [8.709,16.113]|65.53|
|demo4|A2|100|40.441/34.580/14.364/42.543|20.217 [14.849,24.786]|71.74|
|demo6|A0|10|14.816/11.387/4.236/14.915|7.150 [5.845,8.094]|66.96|
|demo6|A0|25|23.481/17.081/6.322/23.489|10.759 [8.679,12.456]|62.67|
|demo6|A0|50|32.616/21.318/9.100/32.655|12.218 [11.088,13.110]|51.87|
|demo6|A0|100|46.768/37.602/21.759/48.251|15.843 [14.131,17.813]|59.80|
|demo6|A1|10|13.628/11.585/4.570/13.717|7.015 [5.545,8.353]|76.69|
|demo6|A1|25|21.571/18.125/7.163/21.564|10.962 [8.275,13.408]|76.12|
|demo6|A1|50|29.916/21.209/11.243/29.947|9.966 [7.782,12.018]|53.28|
|demo6|A1|100|45.085/37.479/22.400/46.312|15.080 [11.757,18.327]|63.06|
|demo6|A2|10|12.829/12.116/3.304/12.900|8.812 [6.846,10.631]|91.83|
|demo6|A2|25|20.643/18.636/5.440/20.640|13.196 [11.021,15.063]|86.82|
|demo6|A2|50|29.282/25.117/9.024/29.294|16.092 [13.882,18.313]|79.39|
|demo6|A2|100|44.829/42.267/20.469/46.250|21.798 [17.909,25.107]|84.55|

### 原48窗/6parent单独对照
|组|前8/H50 L1|F1±5/±10|状态准确率|真事件窗口数/物理去重|
|---|---|---|---|---|
|A0|0.011367/0.058894|0.2119/0.2452|0.8983|13/13|
|A1|0.011936/0.062770|0.2333/0.4250|0.9483|13/13|
|A2|0.007999/0.039907|0.3000/0.4667|0.9587|13/13|

|组|h|GT / pred N1 / no-traj / oracle skill %|N1−no pp [CI]|
|---|---|---|---|
|A0|10|13.147/9.903/4.160/13.257|5.743 [3.709,7.231]|
|A0|25|21.918/14.424/6.495/21.969|7.929 [4.730,10.053]|
|A0|50|30.354/18.651/8.970/30.406|9.681 [6.319,11.761]|
|A0|100|44.267/35.127/17.869/45.984|17.259 [11.788,21.287]|
|A1|10|12.014/9.613/4.243/12.113|5.370 [3.555,6.588]|
|A1|25|20.505/15.175/7.939/20.551|7.236 [3.649,9.828]|
|A1|50|27.864/17.829/11.656/27.917|6.173 [2.366,9.164]|
|A1|100|43.999/37.038/20.413/44.830|16.625 [11.297,20.607]|
|A2|10|11.034/10.264/3.505/11.114|6.759 [4.201,8.731]|
|A2|25|19.741/17.933/6.223/19.776|11.710 [8.639,14.718]|
|A2|50|27.607/23.370/9.096/27.619|14.275 [8.971,17.983]|
|A2|100|42.725/40.412/17.742/44.663|22.670 [15.343,28.517]|

### 原48重放核对
{"A0": {"max_action_metric_abs_difference": 0.0, "max_world_mse_abs_difference": 0.0, "events_identical": true}, "A1": {"max_action_metric_abs_difference": 0.0, "max_world_mse_abs_difference": 0.0, "events_identical": true}, "A2": {"max_action_metric_abs_difference": 0.0, "max_world_mse_abs_difference": 0.0, "events_identical": true}}
SUMMARY.json另含原48完整指标、各层完整CI、逐关节/逐步夹爪准确率、全部世界skill CI与保留率CI。A0/A1/A2/evaluation.json为逐窗结果与事件标识，evaluator.py为实际执行函数。

### 资源与审计
{"status": "complete", "brt": "2026-11-15T05:43:03.302946-03:00", "GPUh_upper": 0.07446654590079561, "new_bytes": 9301629, "checkpoint_unchanged": true, "feature_extraction_bytes": 0, "decision": {"joint_by_group": {"A0": false, "A1": false, "A2": true}, "world_by_group": {"A0": true, "A1": true, "A2": true}, "gripper_by_group": {"A1": true, "A2": true}, "gripper_fix_effective": true}, "report": "/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/notes/e102x_extended_eval_20261115.md", "segments": [{"group": "A0", "rc": 0, "elapsed_s": 44.152277566026896, "upper_s": 74.1522775660269}, {"group": "A1", "rc": 1, "elapsed_s": 21.157130155712366, "upper_s": 51.157130155712366}, {"group": "A1", "rc": 0, "elapsed_s": 44.84271585615352, "upper_s": 74.84271585615352}, {"group": "A2", "rc": 0, "elapsed_s": 37.92744166497141, "upper_s": 67.92744166497141}], "eval_source_unchanged": true}
本轮仅推理，无训练、无阈值拟合、无选点，无dev模型语言。E93 24个不可评parent仍不可评：18 metadata-only、5时钟合同缺失、1仅74帧不足既有窗口；不扩大为51parent结论。+100缺1parent已明确分母。未完成：无；单seed不估seed变异。世界条件判据与关节判据通过也不等于整套策略/夹爪或实机部署已被验证。需要用户决定：无，本轮不自动部署。

## 最终中文结论与用户待决事项
(a) 共49个窗口真事件，按canonical/爪/绝对起点/方向去重为48个物理事件。A1−A0 F1±10=0.15029，95%CI=[0.028327387632943193, 0.27422505478061027]；A2−A0=0.14112，95%CI=[0.007904441237774583, 0.2776028806584362]，两者均不跨0且假切换均值下降，满足原预注册夹爪判据。±5两组CI仍跨0，不能推广为±5显著。
(b) A2相对A0的gt_action世界skill下降仍存在，+10/+25/+50/+100分别−2.012/−2.489/−2.649/−2.284pp，各95%CI均完全小于0（具体CI见前表）。不得把动作改善等同于GT条件世界头全面改善。
(c) A2前8/H50关节L1改善分别0.001524/0.013329 rad，95%CI下界分别0.000611/0.008386，均>0。A2在+25/+50/+100的N1−no-traj分别12.206/14.413/21.312pp，CI下界均>0，保留率84.15/74.17/77.72%。关节及世界可部署条件判据仍成立；此处只指预注册离线条件，不构成实机整策略部署验证。
原48窗三组重放：动作指标差0、事件完全相同、原有+25/+50/+100世界MSE差0。新+10沿同一世界合同补评。
未完成项：无可恢复项遗留。覆盖限制：+100有3窗/1parent因长度不足无合法目标，仅报告208窗/26parent；E93其余24个冻结parent继续按既有不可评原因排除。无需用户追加决定；默认基线仍按用户决定为A2，不自动部署。
资源：全过程含失败和每段30s余量≤0.074467GPUh，显著低于0.5；新增特征0，数据与checkpoint只读。共享日志已追加。
最终审计落盘约8.89MiB；三组checkpoint SHA、原代码eval/model/data SHA均未变化；4个GPU段状态均closed。
