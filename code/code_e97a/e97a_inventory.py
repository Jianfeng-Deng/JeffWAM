from e97a_core import *
root=Path('/mnt/workspace/dengjianfeng/data/camera_calibration_20261015')
items=[]
for p in sorted(root.rglob('*.json')):
 if not any(t in p.name.lower() for t in ('calibration','extrinsic','transform','intrinsic')):continue
 try:
  d=json.loads(p.read_text());items.append(dict(path=str(p),bytes=p.stat().st_size,keys=list(d)[:25],robot_hint='tianji' if 'tianji' in str(p) or 'CP28563000JZ' in p.read_text() else 'not_assigned',serial=d.get('serial'),status=d.get('status')))
 except Exception as e:items.append(dict(path=str(p),error=str(e)))
params=P/'bench/piper_bench/assets/params_piper_description_fit_d.npy';pp=np.load(params)
cal=dict(queried_root=str(root),json_candidates=len(items),inventory=items,piper=dict(fk_path=str(P/'bench/piper_bench/fk.py'),urdf=str(fk.URDF),urdf_sha256=sha(fk.URDF),params=str(params),params_sha256=sha(params),focal_px=float(pp[0]),visual_link6_offset_m=float(pp[1]),geometry_offset_used_m=.1358,scope='legacy demo4 top-camera fitted camera-from-left/right-base Rodrigues and translations; cx320 cy240 raw640x480; no independent metric approval; do not transfer demo3/demo6',certified_pixel_gt=False),calibration_conclusion='camera_calibration_20261015 is Tianji project, distinct top serial CP28563000JZ; Piper sampled top sdk262322075245 / usb343323021050. No verified Piper top extrinsic found there. Existing demo4 legacy fit only; full train source manifest calibration_id often null; no projection implemented.')
save('calibration_inventory.json',cal)
paths=[]
for code in ('code_e82','code_e86','code_e88','code_e90','code_e93','code_e96a'):
 paths.extend((T/code).rglob('*.py'))
paths.extend([E93/f'ridge_h{h}_coefficients.npz' for h in H]);paths.extend([E93/'dev/windows_full.json',E93/'dev/captures_full.json',P/'bench/piper_bench/fk.py',fk.URDF,params,T/'results/e96a_m1_p1_20261114_01/ckpt_05000.pt'])
save('protected_hashes.json',{str(p):sha(p) for p in paths})
with (T/'notes/e97a_robot_env_decomp_20261114.md').open('a') as f:
 f.write('\n### 标定只读盘点\n`camera_calibration_20261015` 是天机几何项目，README 明示天机，顶视 serial CP28563000JZ，与 Piper 的 RealSense 顶视 sdk262322075245 / usb343323021050 不同；不能混用。枚举 '+str(len(items))+' 个相关 JSON，清单见 calibration_inventory.json。找到 Piper demo4 旧 URDF 顶视拟合参数（f='+str(pp[0])+', d='+str(pp[1])+')，但 E84 30 帧投影审计中位误差37.4px、p90=71.7px，未认证精确像素GT；demo3/demo6 无认证外参。没有实现投影。\n')
 f.write('\n### 能量/剩余指标预先约定（dev 输出前）\ny 是固定归一化真实残差，s=R1(q)；机器人可解释份额定义为 1−E||y−s||²/E||y||²（允许负值），剩余份额为 E||y−s||²/E||y||²。另报 s² 与 2s·(y−s) 交叉项，禁止把非正交预测能量硬当百分比分区。减法剩余上的模型输出为 m−s，零剩余复制等价 oracle R1 基线；其误差恒等于原误差，故该口径检验能否超过 R1，不能伪装成独立的环境像素定位。R2−R1 直接给出图像条件对状态剩余的增益。\n正交补充：各格用训练R1系数行与截距的输出span构造Q，SVD数值秩阈值为 max(shape)×float32eps×最大奇异值（预固定，非dev选择）。y_robot=QQᵀy，y_env=(I−QQᵀ)y；对copy/R0/E96a同样投影再评分，能量和MSE改进精确相加。此span可能包含与机器人相关的物体运动、相机运动、阶段及背景方向；span内真实能量也不全是状态模型能准确预测的能量。不得把它当物理机器人/环境真值。\n')
print(json.dumps(dict(status='completed',calibration_candidates=len(items),protected_files=len(paths))))
