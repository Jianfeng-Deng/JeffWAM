import json
from pathlib import Path
T=Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
R=T/"results/e99b3_20261115_01"
OUT=T/"results/e116p3_20261117_01"
spec=json.loads((R/"spec.json").read_text())
eng_names=["Approach the selected zip tie","Lift the selected zip tie from the bin","Hand off the locking head","Reach for the free tail","Bend the tail back and align it with the locking head","Feed the tip through and confirm it exits","Regrasp the emerged segment and pull a loop","Lower and release the finished tie","Withdraw and decide whether to start another tie","Recover from a failed grasp or insertion"]
eng_start=[
"An empty right gripper moves toward an upright strap in the bin or basket.",
"The selected strap lies between the open fingers of the right gripper.",
"The right hand moves a held strap near the locking head toward the open left gripper.",
"The left hand holds the locking head while the empty right gripper moves toward the hanging lower strap.",
"The right hand moves the held free end from its hanging position back toward the locking head in the left hand.",
"The tip begins to advance into the opening near the locking head.",
"A segment has emerged from the exit and the right gripper moves to the exit side of the locking head.",
"The held loop is lowered toward the tabletop or into the basket.",
"The target has been released and both arms move away from the drop point.",
"A missed grasp, slip, lost connection, or withdrawal and reinsertion of the same tie is visible."]
eng_end=[
"The selected strap is between the two fingers, with its locking head above the fingertips and not lifted with the gripper.",
"The held target moves with the right gripper away from the bin while neighboring straps remain inside.",
"The left gripper supports the locking head after the right gripper leaves its original grip, and the target does not fall.",
"The free end moves with the right gripper, with the tip exposed beyond the fingers.",
"The tip is aligned with the opening in the wrist view, with nearby timing and the auxiliary view supporting depth alignment.",
"A new visible segment appears on the exit side while the locking head remains supported by the left hand.",
"After pulling, the exit segment remains through the locking head and a loop remains; the right gripper can unload.",
"The receiving surface supports the newly placed target, both grippers open, and neither pulls the target.",
"Both grippers are clear of the finished tie, and a visible next tie is selected or the batch enters a wait.",
"The grasp or alignment relation is restored for another attempt, or the target is abandoned and released onto a visible receiving surface."]
stages=[]
for i,s in enumerate(spec["stages"]):
 stages.append({"id":s["id"],"name_zh":s["name_zh"],"visible_start_zh":s["visible_start_zh"],"visible_end_zh":s["visible_end_zh"],"name_en":eng_names[i],"visible_start_en":eng_start[i],"visible_end_en":eng_end[i]})
preds=[{k:p[k] for k in ("id","stage","question_en","question_zh","primary_camera")} for p in spec["predicates"]]
lock={"locked_before_dev":True,"primary_stage_text_group":"start_plus_end","primary_stage_language":"en","primary_camera_fusion":"mean of three per-camera logits","image_preprocess":"code_e86/e86/vision.py PREPROCESS, 224 aspect-preserving resize + edge-center padding; no crop","text_tokenization":"padding=max_length, max_length=64, truncation=False","embedding":"SiglipModel get_image_features/get_text_features; L2 normalized; image-text cosine multiplied by model logit_scale + logit_bias then sigmoid","stage_text_sensitivity_groups":["name","start","start_plus_end"],"stage_text_languages":["en","zh"],"stage_count":10,"predicate_count":21,"predicate_metric_primary_camera":True,"P073":"fixed abstention; excluded from zero-shot AUROC macro","stages":stages,"predicates":preds}
p=OUT/"text_lock.json"
assert not p.exists()
p.write_text(json.dumps(lock,ensure_ascii=False,indent=2)+"\n")
report=T/"notes/e116p3_language_alignment_2026-11-17.md"
with report.open("a") as f:
 f.write("\n## 2. 锁定文本（dev 评估前）\n\n")
 f.write("主阶段文本固定为 **起始+结束描述，英文**；相机融合固定为三相机 logit 均值。选择理由是该组完整描述阶段可见关系而非仅标签名；本文不根据 dev 调整。另并列报告阶段名/起始描述 × 中英全部敏感性。英文翻译一次锁定。谓词主评估固定用英文 question_en 与 primary_camera；中文为敏感性。P073 固定弃权。完整锁定见 `results/e116p3_20261117_01/text_lock.json`。\n\n")
 f.write("|阶段|中文名|English name|起始中文|Start description|结束中文|End description|\n|---|---|---|---|---|---|---|\n")
 for s in stages:
  f.write("|"+"|".join(s[k].replace("|","/") for k in ("id","name_zh","name_en","visible_start_zh","visible_start_en","visible_end_zh","visible_end_en"))+"|\n")
 f.write("\n谓词问句按规格原文锁定，文件含 21 项中英文本及 primary_camera；P073 固定弃权。\n")
print("LOCKED",len(stages),len(preds),p.stat().st_size)
