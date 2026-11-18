import collections
import datetime
import hashlib
import json
import math
import pathlib
import subprocess

T = pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu')
R = T / 'data/e84_piper_align_v1'
RUN = 'e84p_20261113_021548'
P = R / 'probes' / RUN
REPORT = T / 'notes' / f'e84_P_{RUN}.md'

def sha(p):
    h = hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def save(name, obj):
    (P / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def append(s):
    with REPORT.open('a') as f:
        f.write('\n' + s + '\n')

if json.loads((P/'progress.json').read_text()).get('status') in {'stopped_no_free_gpu','stopped_after_initial_no_free_gpu'}:
    print('Run finalized; inspect progress.json. Do not regenerate receipts or duplicate logs; use a new resume entry point.')
    raise SystemExit(0)

now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()
receipts = {}
for name in ['READY_split.json', 'READY_fk_gt.json', 'READY_pixel_gt.json']:
    f = R / name
    obj = json.loads(f.read_text())
    receipts[name] = {'sha256': sha(f), 'content': obj}
save('accepted_receipts.json', receipts)
pixel_ready = receipts['READY_pixel_gt.json']['content']
verified = []
for name, entry in pixel_ready['files'].items():
    assert sha(entry['path']) == entry['sha256'], name
    verified.append(name)
split_ready = receipts['READY_split.json']['content']
for entry in split_ready['files']:
    assert sha(entry['path']) == entry['sha256']
split = json.loads(pathlib.Path(split_ready['files'][0]['path']).read_text())
assert split['source_interface_revision'] == 3
assert split['model_results_seen'] is False
sampling = json.loads((R / 'pixel_gt/sampling.json').read_text())
assert sampling['source_split_sha256'] == split_ready['files'][0]['sha256']
rows = [json.loads(s) for s in (R / 'pixel_gt/pixel_gt.jsonl').read_text().splitlines()]
assert len(rows) == 150 and len({r['id'] for r in rows}) == 150
allowed = {r['canonical_id']: r for r in split['captures']}
for r in rows:
    assert split['parents'][r['parent_id']] == r['probe_split']
    assert allowed[r['capture']]['parent_component'] == r['parent_id']

counts = {}
for key in ['L', 'R', 'TIE', 'HEAD', 'TIP']:
    c = collections.Counter()
    for r in rows:
        a = r['annotations'][key]
        visible = a['visibility'] == 'visible'
        coords = a['point_xy'] is not None or a['bbox_xyxy'] is not None
        c['total'] += 1
        c['visible'] += visible
        c['ambiguous'] += bool(a['ambiguous'])
        c['low_all'] += a['confidence'] == 'low'
        c['visible_low'] += visible and a['confidence'] == 'low'
        c['main_eligible'] += visible and coords and not a['ambiguous'] and a['confidence'] != 'low'
        c['point_flag_true'] += bool(a['eligible_for_point_probe'])
    counts[key] = dict(c)

groups = collections.defaultdict(list)
for r in rows:
    groups[r['capture']].append(r)
gaps = []
pair_counts = {'consecutive_source_frames': 0, 'within_0_25s': 0, 'within_1s': 0}
for cid, rr in groups.items():
    rr.sort(key=lambda r: r['source_frame_index'])
    for i, a in enumerate(rr):
        for b in rr[i+1:]:
            dt = b['exposure_timestamp'] - a['exposure_timestamp']
            df = b['source_frame_index'] - a['source_frame_index']
            assert dt > 0 and df > 0
            pair_counts['consecutive_source_frames'] += df == 1
            pair_counts['within_0_25s'] += dt <= .25
            pair_counts['within_1s'] += dt <= 1
            common = [k for k in ['L','R','HEAD','TIP'] if all(
                x['annotations'][k]['point_xy'] is not None and
                x['annotations'][k]['confidence'] != 'low' and
                not x['annotations'][k]['ambiguous'] for x in [a,b])]
            gaps.append({'a':a['id'],'b':b['id'],'capture':cid,'dt_s':dt,
                         'source_frame_gap':df,'common_trusted_point_keys':common})
gaps.sort(key=lambda x: x['dt_s'])
radii = {}
for cohort in ['demo4','demo6_early','demo6_native']:
    shape = next(r['resolution'] for r in rows if r['stratum'] == cohort)
    noise = pixel_ready['noise_estimate'][cohort]
    p90 = max(noise['two_pass']['p90'], noise['third_pass']['p90'])
    patch = max(shape) / 14
    r = max(3*p90, patch)
    radii[cohort] = {'noise_p90_px':p90,'patch_edge_original_px':patch,
                      'r_primary_px':r,'r_sensitivity_px':2*r}
audit = {'run_id':RUN,'timestamp_brt':now,'pixel_ready_files_hash_verified':verified,
         'frames':len(rows),'parents':len({r['parent_id'] for r in rows}),
         'split_frames':dict(collections.Counter(r['probe_split'] for r in rows)),
         'split_parents':{s:len({r['parent_id'] for r in rows if r['probe_split']==s}) for s in ['probe-train','probe-val']},
         'strata':dict(collections.Counter(r['stratum'] for r in rows)),
         'target_counts':counts,'pair_counts':pair_counts,
         'nearest_pair':gaps[0] if gaps else None,'pairs':gaps,'proposed_fixed_radii':radii,
         'radius_note':'For inherited full-FOV aspect-preserving edge-pad 224 preprocessing only; fixed without model scores. 2r sensitivity; no validation optimization.',
         'pixels_or_projection_used_as_truth':False,'raw_images_read':0,'dev_test_samples_read':0}
save('pixel_metadata_audit.json', audit)
append('## 交付及字段核验完成\n\n'+f'BRT {now}。G回执全部{len(verified)}个入口SHA256通过，split SHA及150帧parent/capture归属一致。'+
       '仅读取交付元数据和近似标注JSON；未读取原始图像、dev/test样本或投影运动数据。\n\n'+
       f'帧数/parent：{len(rows)}/{audit["parents"]}；split帧数 {audit["split_frames"]}；split parent {audit["split_parents"]}。\n\n'+
       '|对象|可见|可见且low|ambiguous|主分析可用|\n|---|---:|---:|---:|---:|\n'+
       '\n'.join(f'|{k}|{v["visible"]}|{v["visible_low"]}|{v["ambiguous"]}|{v["main_eligible"]}|' for k,v in counts.items())+
       '\n\n主分析排除low、ambiguous及无坐标项；low分层只作敏感性诊断。L/R是夹口中心近似点，不能充当整臂像素掩码；TIE仅紧框，inside/outside只能称框代理AUROC。right/left arm查询需单列夹爪点代理，整臂真值N/A。HEAD/TIP用点邻域AUROC，均非真实分割。')
append('## 相邻标注检查与预定半径\n\n'+f'同capture穷举配对：{pair_counts}。最近配对：`{json.dumps(audit["nearest_pair"],ensure_ascii=False)}`。'+
       '\n\n本次不使用FK投影造像素运动标签，未计算光流弱真值；画面像素运动方向记为“无合适相邻真值，未评估”（如存在少量邻对，仅保留数量审计，不外推全体）。机器人系3D FK方向/幅度与动作反事实不依赖标定，仍属于已授权续跑范围，但当前受GPU门槛阻塞。\n\n'+
       '未看模型结果前固定后续半径：各层r=max(3×max(两遍P90,第三遍P90),原图一个patch边长)，另报2r。沿用224保比例edge-pad时，原图patch边长=max(W,H)/14。不是根据probe表现选阈值。\n\n'+
       '|层|一个patch(px)|r(px)|2r(px)|\n|---|---:|---:|---:|\n'+
       '\n'.join(f'|{k}|{v["patch_edge_original_px"]:.2f}|{v["r_primary_px"]:.2f}|{v["r_sensitivity_px"]:.2f}|' for k,v in radii.items()))

gpu_raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpu = []
for line in gpu_raw.splitlines():
    idx, uuid, mem, util = [x.strip() for x in line.split(',')]
    gpu.append({'index':int(idx),'uuid':uuid,'memory_mib':int(mem),'utilization_percent':int(util),
                'eligible':int(mem)<512 and int(util)<5})
save('gpu_snapshot.json', {'timestamp_brt':now,'devices':gpu,'eligible_count':sum(x['eligible'] for x in gpu)})
append('## 像素真值局限与结论降级\n\n150帧来自同一模型，非独立人工或绝对真值；第三遍68点重复差中位2.83px、P90 5.39px，native两遍P90 15.90px。demo4/early/native标定验证中位26.90/9.07/18.97px，全部NO-GO，禁止当自动像素真值。可见低置信和身份歧义按上表处理；小尖端、相机分辨率和同parent相关性限制统计力。后续只作粗粒度诊断，按parent bootstrap 2000次95%CI，并分报probe-train与probe-val；不将150帧均称验证集。像素结果不能支持精细穿带验收或选择checkpoint。\n\n当前无模型结果，左右是否失败不可判定；未触发LoRA，也没有依据推荐更强VLM。阶段一方案暂不在“冻结即可/需要LoRA/更强VLM”中强行选择。')

status = 'stopped_no_free_gpu' if not any(g['eligible'] for g in gpu) else 'stopped_after_initial_no_free_gpu'
progress = json.loads((P/'progress.json').read_text())
progress.update(status=status,updated_brt=now,pixel_metadata_audit_complete=True,
                gpu_hours=0,probe_gpu_hours=0,lora_gpu_hours=0,model_frames=0,train_runs=0,
                awaiting='eligible GPU; resume revised experiments with a new resource check',
                lora_trigger='undetermined_no_model_scores',raw_images_read=0,dev_test_samples_read=0)
save('progress.json',progress)
append('## 最终资源状态\n\n'+f'BRT {now} 再核对：合格卡{sum(g["eligible"] for g in gpu)}/8，最低显存占用{min(g["memory_mib"] for g in gpu)}MiB。'+
       '首轮也为0/8；依据用户修订第5条“没有空闲 GPU 就停下报告”，停止，不启动两小时等待、CPU回退、特征提取或训练。GPU累计0 GPUh，无新后台任务，不操作他人进程。旧run及F/G目录未改。续跑无需新增授权，只需在后续执行时重新核对资源。')
summary = T/'notes/e84_alignment_summary_20261113.md'
assert not summary.exists(), 'do not overwrite existing summary'
summary.write_text(f'''# E84 对齐验证续跑：资源阻塞

ID `{RUN}`；BRT {now}。已接受用户授权的150帧模型近似像素标注范围；本次因无合格空闲GPU停止。**实验尚未完成，GPU 0 GPUh。**

|对齐|冻结零样本|冻结特征探针|LoRA|
|---|---|---|---|
|语言↔图像|未运行：GPU门槛|未运行|不可判定，未触发|
|语言↔画面像素运动|无合适相邻真值，未评估|未评估|未评估|
|语言↔机器人系FK 3D运动|已授权，GPU阻塞|已授权，GPU阻塞|未触发|
|语言↔动作反事实|需条件预测器|已授权，GPU阻塞|未触发|

没有pointing/AUROC、模型对照差距或bootstrap CI；不将“未运行”当作失败。24类随机top1/top3理论值4.17%/12.5%，8向12.5%、3幅度33.3%，不代表实测。左右失败不可判定，不据此触发LoRA；冻结/LoRA/更强VLM选择仍需实验。

## 已完成核验

G回执{len(verified)}个入口SHA通过；150帧{audit['parents']}parent，probe-train/val=110/40帧，val=32parent。低置信/ambiguous字段及同capture时间邻对已统计，详见主报告和pixel_metadata_audit.json。相邻对统计：{pair_counts}。未使用FK像素投影或光流。

## 像素真值局限

同一模型近似标注，第三遍中位2.83px、P90 5.39px，native两遍P90 15.90px；不是绝对误差界。三层标定验证中位26.90/9.07/18.97px，均NO-GO。150帧中110为probe-train，40为probe-val，不能冒充150帧独立留出；无demo3、腕视角标注或整臂mask。TIE紧框只能用于框代理AUROC，夹爪/锁头/尖端为点邻域评估。后续需分报low/ambiguous与parent bootstrap，并披露小样本统计力。半径预定r与2r（640宽r=45.71px，1280宽r=91.43px，沿用保比例224预处理），不按模型表现调节。该空间粒度不足以确认精细定位/穿带结论。

## 编码器与语言来源

继承旧run已核验结论，非本次重新运行：本地SigLIP2 base-patch16-224，375187970参数，视觉/文本双塔，196×768 patch，文本64token，非NaFlex。逐patch singleton-MAP后与文本余弦；MAP原始attention不依赖文本，逐patch池化偏离全局训练分布，分数不是分割概率。

E83-A已在旧run后更新，旧run的107条/106parent语言统计不可当作当前全量统计。本次因GPU停止未重跑文本审计。旧冻结快照中E83-B为94条/94parent、8种完整任务指令、1种最短改写；FK旧已审计435404行/243parent，18393种精确句、去数字471种。text↔parent映射保存在旧run；本次仅引用历史快照，不称重新实算。FK局部方向/幅度文本比少数完整任务指令更适合提供局部语言条件；须防未来描述作为部署输入泄漏。最终续跑应重算当前E83与FK信息量。

## 断点与未完成项

按修订第5条无空闲卡即停止，未启用旧2h CPU回退；GPU 0 GPUh，模型帧0、正式训练0。待合格空闲卡后：新run冻结特征→150帧图文诊断与对照/CI→机器人系运动探针→语言条件位移反事实→条件LoRA。无需用户再次批准既定范围，阶段一取舍待结果。

主报告：`{REPORT}`；进度：`{P/'progress.json'}`；输入与字段审计：`{P/'pixel_metadata_audit.json'}`；GPU记录：`{P/'gpu_snapshot.json'}`。旧run `e84p_20261112_153338` 未覆盖。
''')
with (T/'docs/experiment_log/EXPERIMENT_LOG.md').open('a') as f:
    f.write(f'\n### {now} BRT — {RUN} E84-P 续跑资源检查\n\n目标：按2026-11-13修订续跑图文/FK运动/动作反事实。改动：新run报告骨架、近似GT入口哈希/字段/邻帧审计、GPU快照与progress；不改F/G或旧run。命令：`python3 code_e84/{RUN}/resume_audit.py`，`nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader`。目录：`{P}`，主报告`{REPORT}`，汇总`{summary}`。结果：150帧metadata核验通过；合格GPU 0/8（首轮为0），GPU 0 GPUh，未读原图/未训练。结论：依据用户修订第5条停止，像素标定仍NO-GO；不据此触发LoRA或选择阶段一。状态：stopped_no_free_gpu，完整实验未完成。\n')
with (T/'notes/worklog.md').open('a') as f:
    f.write(f'\n- {now} BRT {RUN}：150帧近似GT入口SHA/字段/时间邻对核验完成；GPU 0/8合格，依修订停止，0 GPUh；实验未运行，续跑无需新增授权。汇总 notes/{summary.name}。\n')
print(json.dumps({'run_id':RUN,'status':status,'frames':len(rows),'parents':audit['parents'],
                  'target_counts':counts,'pairs':pair_counts,'nearest':audit['nearest_pair'],
                  'eligible_gpu_count':sum(g['eligible'] for g in gpu),'summary':str(summary)},ensure_ascii=False))
