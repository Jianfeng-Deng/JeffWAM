"""Append verified deliverables and logs; no model/source mutations."""
import json,csv,os,fcntl,re,subprocess
from pathlib import Path
import numpy as np
from e88.common import *
from e88.train import source_hash
run=ART/'smoke_03'
def read(p):return json.loads(Path(p).read_text())
def fmt(x,n=4):return 'NA' if x is None else f'{x:.{n}f}'
def table(headers,rs):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rs])
if (ART/'DELIVERY_READY.json').exists():raise FileExistsError('Already finalized')
assert read(ART/'pipeline_02_status.json')['status']=='completed'
assert read(ART/'after_pipeline_status.json')['status']=='completed'
assert read(run/'conditional_inference_01/READY.json')['status']=='ready'
assert read(run/'status.json')['status']=='completed'
ledger=read(ART/'gpu_ledger.json');assert not ledger.get('active');assert ledger['elapsed_s']<=9000
assert read(run/'contract.json')['code_sha256']==source_hash()
for name in ('P1','P3_short'):assert read(run/(name+'_external_resume.json'))['status']=='passed'
for name in ('P1','P3_none','P3_short','P3_long'):
 r=read(run/(name+'_resume_check.json'));assert r['loss_equal'] and r['parameters_equal']
contract=read(run/'contract.json');availability=read(ART/'cpu_audit_01/availability.json');cov=read(ART/'cpu_audit_01/stage_coverage.json')['coverage'];grip=read(ART/'cpu_audit_01/gripper_cpu.json');cache=read(ART/'cpu_audit_01/cache_reads.json');extra=read(ART/'cpu_audit_01/additional_checks.json');results=read(run/'stage_results.json');profile=read(run/'profile_01/READY.json');conditions=read(run/'profile_conditions_01/READY.json');t1=profile['results']['P1']['mean_s'];rates={'short':profile['results']['P3']['mean_s'],**{k:v['mean_s'] for k,v in conditions['results'].items()}};raw=(t1*5000+rates['long']*10000)/3600;estimate=dict(P1_5000_GPUh=t1*5000/3600,P3_10000_GPUh_by_condition={k:v*10000/3600 for k,v in rates.items()},total_GPUh=raw,with_25pct_margin_GPUh=raw*1.25,paired_P3_GPUh=(t1*5000+rates['long']*20000)/3600*1.25,three_language_conditions_one_seed_GPUh=(t1*5000+sum(rates.values())*10000)/3600*1.25,primary_condition='long',scope='Measured full pools batch32 all six horizons plus random, chain 1/4; three actual text conditions. Excludes queue/evaluation/cache rebuild.');atomic(run/'formal_estimate.json',estimate);chain=read(run/'chain_diagnostics.json');gate=read(run/'adapter_gate.json');gpu=ledger['elapsed_s']/3600
lines=['\n## E88-W 最终交付（'+brt()+' BRT）','状态：完成有界工程验证；未启动正式训练。唯一新包code_e88。原code_e86/e82、数据入口、E83和缓存只读，无下载、无LoRA。', '### A 夹爪最终判定与曲线','已修复signed-current复制基线的负值截零口径错误；未发现动作目标/反归一化/mm换算/next-frame索引错误。夹爪锚点为0符合signed absolute定义，不能用关节式q残差偷偷替换。2/14归一化动作维的损失份额与其物理mm尺度都已核查。约16mm的模型偏差主要是烟雾训练/流目标尚未学好；需要完整FM收敛实验，不能由独立小头过拟合证明已解决。','该极小头误差是对浮点训练目标的数值拟合，不代表传感器或机械精度。', '独立64窗夹爪头188步CPU过拟合H50='+fmt(grip['head_only_overfit']['H50_mm'],6)+'mm；同窗signed复制H50='+fmt(grip['train64']['copy_signed_current']['H50_mm'],6)+'mm。无dev拟合。']
rows=[]
for file,label in [('legacy_dev_gripper.json','v3 dev48'),('dev_P3_none.json','v3.1 none dev48'),('dev_P3_short.json','v3.1 short dev48'),('dev_P3_long.json','v3.1 long dev48'),('legacy_train64_gripper.json','v3 train64'),('v31_train64_gripper.json','v3.1 long train64')]:
 d=read(run/file)
 for mode,v in d['summary'].items():rows.append([label,mode,fmt(v['first8_grip_mm']),fmt(v['H50_grip_mm']),fmt(v['first8_joint_rad'],6),fmt(v['H50_joint_rad'],6)])
lines.append(table(['样本/模型','条件','grip前8/mm','gripH50/mm','joint前8/rad','jointH50/rad'],rows));lines.append('完整50步曲线：artifacts/smoke_03/gripper_curves.csv、gripper_curves.png；dev保持原6parent/48窗、parent等权，真实next-frame follower反馈，未加模型语言或重选窗口。')
lines+=['### B 六时域全池与分层','每个未来目标：按anchor+h真实timestamp，在±2原行里选每路最近曝光，误差≤1/60秒；连续segment、source mask、V3独占子区间全部满足，三个相机对应原行都已有缓存。动作/标签不平移。']
rows=[]
for s,byh in availability.items():
 base=107057 if s=='P1' else 34777;np0=242 if s=='P1' else 145
 for h,d in byh.items():rows.append([s,h,d['windows'],d['parents'],fmt(100*d['windows']/base,2)+'%',fmt(100*d['parents']/np0,2)+'%'])
lines.append(table(['池','+帧','有效窗','parent','窗口保留','parent覆盖'],rows));lines.append('正式最长时域建议+100（P3保留62.647%、parent100%）；+200/+300未达60%窗口门槛，仅保留工程诊断。正式默认链25+25→50、50+50→100；100+100→200须在单独放行长时域配置后使用。')
rows=[]
for s,byh in availability.items():
 for h,d in byh.items():
  for st,v in d['strata'].items():rows.append([s,h,st,v['windows'],str(v['parent_coverage'])+'/'+str(v['total_parents']),v['total_windows']])
lines.append(table(['池','+帧','分层','有效窗','parent有效/总','总窗'],rows));rows=[]
for h,d in cache.items():rows.append([h,'/'.join(map(str,d['required_unique_source_valid_camera_frames'])),'/'.join(map(str,d['already_cached_camera_frames'])),'/'.join(map(str,d['missing_camera_frames']))])
lines.append(table(['+帧','源有效所需相机帧 高/左/右','已缓存','若扩覆盖仍缺'],rows));lines.append('此表是P1+P3去重后的原行读取需求，合法窗口数另要求三路同时命中。窗口行表P1_horizon_rows.npz/P3_horizon_rows.npz直接记录每窗每时域三个原行；随机时域逐窗原位检查，不补零冒充有效目标。本轮未扩缓存，新增特征字节0；额外camera source-mask复核失败数='+str(extra['camera_source_mask_failures'])+'。')
lines+=['### C 实现与指标合同','连续正弦物理帧偏移嵌入，固定{10,25,50,100,200,300}+相邻区间随机整数源偏移；不插值图像/标签。短时域每路8细token，长时域每路global1 token；固定W81，带随机最多105。长W只读O/长W，短W读O/A/全部W；长W禁止读A以防间接泄露；O仅读O。确定性余弦只用于短W。','残差y=(未来固定SigLIP2编码−当前固定编码)/每camera/token RMS变化。尺度由P1训练parent等权、每parent每时域最多2窗计算，floor=.05；详见residual_stats.json中原窗/parent列表和contract绑定SHA。随机h同分辨率内插值尺度，51..99暂沿用100尺度；无开发/测试统计。','768→256投影会丢失满维噪声；在e88加入v=(clean_residual_head−noisy_residual)/(1−tau)显式满维噪声旁路（分母floor=.01），clean head零初始化精确退化为复制当前编码。smoke_02保存无旁路控制及源码。短/长均仍是原FM监督，长时域不加确定性余弦。默认P1收敛门关闭，正式由外部明确声明；本次显式engineering_chain_probe仅验证工程ramp。', '世界指标为W-only预测，不带动作/状态。分层parent有交集，不可将各层parent数直接相加。', '每时域分别按有效token平均，再按有效时域平均FM；余弦只短时域。指标为同一归一化残差坐标MSE，copy预测零残差，linear仅用当前与最近有效历史真实编码按每路曝光时间外推。skill_copy=1−model_MSE/copy_MSE；另报skill_linear。负skill表示差于基线，绝对loss不能替代skill。']
lines.append('以下为固定**训练窗工程诊断**，不是开发/测试真值或checkpoint选择。每层至多4窗，优先有+300缓存的窗口；因此是长时域条件子样本，不能代表全池质量。三个P3支路从同一P1初始化各192步，不能作语言效能结论。')
rows=[]
for name in ('P1','P3_none','P3_short','P3_long'):
 d=read(run/('world_'+name+'.json'))
 for h,v in d.items():
  x=v['overall'];rows.append([name,h,x['windows'],x['parents'],fmt(x['model']),fmt(x['copy']),fmt(x['linear']),fmt(x['skill_copy']),fmt(x['skill_linear'])])
lines.append(table(['支路','+帧','窗','parent','model','copy','linear','skill_copy','skill_linear'],rows))
# machine-readable stratified world metrics are full deliverables; include compact per-stratum skill table.
rows=[]
for name in ('P3_none','P3_short','P3_long'):
 d=read(run/('world_'+name+'.json'))
 for h,v in d.items():
  for st,x in v['strata'].items():rows.append([name,h,st,x['windows'],fmt(x['model']),fmt(x['copy']),fmt(x['linear']),fmt(x['skill_copy'])])
lines.append(table(['支路','+帧','层','窗','model','copy','linear','skill_copy'],rows))
matched=read(run/'conditional_inference_01/results.json');rows=[]
for name,byh in matched.items():
 for h,d in byh.items():
  x=d['overall'];rows.append([name,h,fmt(x['model']),fmt(x['copy']),fmt(x['linear']),fmt(x['skill_copy'])])
lines.append('同一固定P3_long阶段末checkpoint、同窗/同噪声的输入条件消融（无优化），避免三训练支路中none同时关闭对齐损失的混杂：')
lines.append(table(['同权重输入条件','+帧','model','copy','linear','skill_copy'],rows))
devworld=read(run/'conditional_inference_01/dev_world_short.json');rows=[]
for h,d in devworld.items():
 x=d['overall'];rows.append([h,x['windows'],x['parents'],fmt(x['model']),fmt(x['copy']),fmt(x['linear']),fmt(x['skill_copy'])])
lines.append('补充独立dev世界参考：固定P3_long阶段末、仅既有人工短指令、原48窗和既有dev缓存，不增模型语言标签/不扩缓存；长时域缺缓存即NA。此表不是语言消融或checkpoint选型。')
lines.append(table(['dev +帧','窗','parent','model','copy','linear','skill_copy'],rows))
lines+=['### D 适配器与链式一致性','适配器独立AttentionPool：真实固定2×4×768→O当前8×256，目标为同帧current(raw).detach()，随后重加入camera/age/kind。64窗拟合、16个不同训练窗校准；门槛NMSE≤.1、cos≥.95，不取模型标注/评估标签。初次'+json.dumps(gate,ensure_ascii=False)+'。','每16步校准复查门；未过门不执行主链。第一跳stop-grad，真实/预测两种重入均走adapter；预测混合概率.2→.8。历史移到a+k，因果可得且已缓存的真实历史保留，其余mask。W-only，不输入状态/动作；直接+2k预测一端stop-grad。第二跳对真实+2k监督+与直接预测一致性；50+50的细输出按camera平均对粗目标，100首跳global显式广播到8格重入，空间细节不可恢复。','当前重入复用legacy窗口构造器，除真实链端点外额外要求shifted+50仍合法；25链靠近段尾会保守少收，尚未优化为最大覆盖。', '链最大权重.1，工程烟雾P1边界128+ramp128、P3起点ramp128，每4更新抽一链、至多2窗、2 Euler。P1烟雾边界只验证调度，**不证明P1已收敛**；正式须先确认P1收敛再开ramp。由于1/4子采样，期望链系数为.025乘W外层权重/模式暴露，不假称每更新都施加.1。','真实固定端点增量可加性：128窗cos≈1.000000、相对误差6.13e-11；这是差分代数恒等式和端点映射检查，不是预测链准确性的证据。最终链诊断（每链至多2个真实训练窗，仅工程探针）：',json.dumps(chain,ensure_ascii=False,indent=2)]
rows=[]
for name,d in results.items():rows.append([name,d['steps'],d['chain_updates'],fmt(d['adapter_mse']),fmt(d['adapter_cosine']),d['adapter_ready'],fmt(d['first16']),fmt(d['last16'])])
lines.append(table(['支路','步数','链更新','adapter NMSE','cos','门开','首16 loss','末16 loss'],rows))
lines+=['### E 阶段/进度与补标','监督来自E83 Tier A原分段序号（不声称语义一致ontology）和段内进度；从无未来噪声的clean W query线性预测，世界内部权重.01。缺标、cutoff缺失或cutoff>anchor全部mask；当前段还要求start≤anchor≤end。阶段辅助不进入输入。']
rows=[]
for s,ds in cov.items():
 for st,d in ds.items():rows.append([s,st,d['windows'],d['any_narration'],d['causal_text'],d['stage_progress'],fmt(100*d['stage_progress']/d['windows'],3)+'%'])
lines.append(table(['池','层','总窗','有叙述','可因果输入历史叙述','当前阶段进度有效','覆盖率'],rows));lines.append('进度分布：'+json.dumps(extra['progress_distribution'],ensure_ascii=False)+'。现有叙述大多cutoff=end，有效监督几乎只在段尾，不能据此声称学到进度。建议先补P3各层的因果前缀阶段/进度（约每段3–5个截止点，优先demo3缺标和所有demo6多轮段内），P1按parent覆盖补；维持原split，不标冻结评估parent。当前覆盖率不足是未完成的数据条件。')
lines+=['### F/G 语言消融、恢复、GPUh与正式估算','none：关闭文本与对齐；short：固定人工短任务指令；long：E83任务长指令+cutoff≤anchor的历史叙述。已有长任务指令缺失窗不伪造补齐；因果叙述只训练parent。dev只用既有人工指令和真实反馈。三支路严格相同P1初始化、P3数据/模式/噪声种子；语言效能需正式步数、多种子和独立评估，当前只是工程链路。','训练none支路沿旧合同同时关闭对齐损失，不能作为纯语言单因素证据；已额外做同一个P3_long固定终点的输入条件消融。正式三支须锁定同样对齐目标/权重和阶段辅助，只改变输入文本；同窗、同噪声、同初始化/步数/种子。', '各层实际文本覆盖（短指令全覆盖；长条件缺项保持mask）：'+json.dumps(read(ART/'language_coverage.json'),ensure_ascii=False), '最终CPU合同：'+(ART/'contracts_06.log').read_text().strip().splitlines()[-1]+'；原73项完整保留，新增35项。新增含真实P1/P3多时域前反传、连续时间、残差还原、长短掩码及间接泄露、短余弦、截止/阶段泄露、链first/direct stop-grad与W-only、真实/预测重入、adapter单侧蒸馏、源缝/子区间/缺缓存、signed基线和模式配置。compileall通过。','P1/P3三语言支路内部恢复下一步loss与全模型digest逐位相同；P1和P3_short独立进程也逐位一致。checkpoint绑定optimizer/sampler/Python/NumPy/Torch/CUDA RNG及全部e86+e88源hash和run contract；不绕过历史checkpoint hash声称原run续训。','总GPU墙钟='+fmt(ledger['elapsed_s'],3)+'秒='+fmt(gpu,6)+'GPUh≤2.5；包括全部停止段、成功烟雾、两次独立恢复、全池profile校准。每段<1GPUh，准入GPU util=0、free≥8192×1.5MiB，实际峰值越界/压力/OOM即停；没有MPS/操作他人进程。账本无active worker。']
rows=[]
profile_rows=dict(profile['results']);profile_rows.update({'P3_'+k:v for k,v in conditions['results'].items()})
for s,d in profile_rows.items():rows.append([s,d['pool_windows'],d['batch'],d['measured_steps'],fmt(d['mean_s']),fmt(d['p50_s']),fmt(d['p99_s']),d['chain_steps'],fmt(d['mean_chain_s']),fmt(d['mean_normal_s']),fmt(d['peak_allocated_mib'],2)])
lines.append(table(['阶段','全池窗','batch','测量步','均值s','P50s','P99s','链步','链均值s','普通均值s','峰值MiB'],rows));lines.append('正式估算（仅训练更新，含本配方新增时域和1/4频率链开销；不含缓存重建、排队、正式评估）：'+json.dumps(estimate,ensure_ascii=False,indent=2))
lines.append('语言正式建议每seed：P1 5000、三支路各P3 10000步，至少3 seeds，固定阶段末、不以模型语言标签选checkpoint。共享P1三语言每seed含25%余量约'+fmt(estimate['three_language_conditions_one_seed_GPUh'],2)+'GPUh，3 seeds约'+fmt(estimate['three_language_conditions_one_seed_GPUh']*3,2)+'GPUh；另计独立真实评估。+100正式上限比本次全六时域可能更便宜；当前估算保留全六时域成本，不在未测情况下宣称降幅。若链改成每步或更多Euler，必须重新profile。')
lines+=['### H 配方A/B与用户决定','默认A：30/30/30/10 + sync/independent中25%概率clean辅助，主辅W取均值。B：25/25/25/25、aux0，`--recipe B`可选但本轮未启动。两者clean暴露期望25%；B的W主更新70%→75%（世界/phase/adapter/chain暴露约+7.14%），W前向85→75/100更新（−11.76%），clean额外A前向10→25/100更新，两者一阶总前向均125；新增链预计B略贵，精确耗时须实测。不用同一loss数值冒充两配方等价。','仍需用户锁定正式A/B口径。建议先接受+100上限并补因果阶段前缀，再安排正式多种子；当前不需要为本轮工程验证再授权。','### I 限制与未完成','正式训练未启动；没有语言有效/成功率/实机控制结论。夹爪完整FM头仍待收敛并优于signed复制；阶段当前监督稀少且大多段尾，需补标；+200/+300现有窗口覆盖不达用户门槛；100首跳粗格广播重入的表示损失、长时域预测分布失配、链滚动稳定性和正式每步链成本仍待验证。世界skill是固定训练条件子样本，dev仅6parent动作参考，不能替代独立世界/语言泛化评估。','交付目录：code_e88；报告本文件；全部原源缓存与同事路径未修改，新增特征缓存0。源码对照'+str(read(ART/'legacy_source_comparison.json')['e86_unchanged_from_copy'])+'。']
report('\n\n'.join(lines))
# Per-step curves in reusable CSV and standalone figure.
curves={}
for file,label in [('legacy_dev_gripper.json','v3 dev48'),('legacy_train64_gripper.json','v3 train64'),('dev_P3_none.json','v31 none dev48'),('dev_P3_short.json','v31 short dev48'),('dev_P3_long.json','v31 long dev48'),('v31_train64_gripper.json','v31 long train64')]:
 d=read(run/file)['summary'];curves[label]=d['action_only']['curve_grip_mm']
 if 'train64' in label and 'signed copy train64' not in curves:curves['signed copy train64']=d['copy_current_signed']['curve_grip_mm'];curves['legacy copy train64']=d['copy_e86_q']['curve_grip_mm']
 if 'dev48' in label and 'signed copy dev48' not in curves:curves['signed copy dev48']=d['copy_current_signed']['curve_grip_mm'];curves['legacy copy dev48']=d['copy_e86_q']['curve_grip_mm']
curves['tiny head train64']=grip['head_only_overfit']['curve_mm']
with (run/'gripper_curves.csv').open('x') as f:
 w=csv.writer(f);w.writerow(['next_feedback_step']+list(curves))
 for i in range(50):w.writerow([i+1]+[v[i] for v in curves.values()])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for label,values in curves.items():
 ax=axes[0] if 'dev48' in label else axes[1];ax.plot(np.arange(1,51),values,label=label,linewidth=1.5)
for ax,title in zip(axes,['Fixed dev: 6 parents / 48 windows','Train 64 windows: model and head-only fit']):
 ax.set(title=title,xlabel='Next-frame feedback step',ylabel='Gripper L1 (mm)');ax.grid(alpha=.25);ax.legend(fontsize=7)
fig.savefig(run/'gripper_curves.png',dpi=180);plt.close(fig)
REPORT.write_text(REPORT.read_text().replace('状态：执行中。','状态：工程验证完成（未启动正式训练）。').replace('待完成。','已完成；详见下方最终交付及证据。').replace('执行中；不以模型标签','工程验证完成；不以模型标签'))
# Include canonical machine-readable refs, total storage and final logs.
size=int(subprocess.check_output(['du','-sb',str(OUT)],text=True).split()[0]);atomic(ART/'storage.json',dict(new_package_bytes=size,new_feature_cache_bytes=0,prior_Piper_cache_inventory_bytes=42164345613,conservative_total_prior_inventory_plus_all_e88_bytes=42164345613+size,limit_bytes=100000000000))
entry=f'\n## E88-W-v31-20261114 — {brt()} BRT\n目标：Piper v3.1多时域残差世界头/链式重入/阶段辅助与夹爪核查。改动：新code_e88；旧包/数据/缓存只读。命令：ssh -o BatchMode=yes <gpu-server>；nohup setsid python -m e88.audit / e88.language / bounded_pipeline_01.py；pytest tests；e88.train smoke_03 --recipe A；e88.replay；e88.profile。目录：{OUT}；报告：{REPORT}。结果：108 CPU合同通过；P1/P3六时域全池统计、建议+100；signed复制基线修复、64窗小头H50 {grip["head_only_overfit"]["H50_mm"]:.6f}mm；P1 160、P3三语言各192步；内部/独立恢复一致；{gpu:.6f}GPUh含失败；正式单链含25%余量估算{estimate["with_25pct_margin_GPUh"]:.3f}GPUh。结论：工程完成；世界/夹爪/语言性能未验收；当前阶段监督P3仅51窗。状态：COMPLETE_ENGINEERING_SMOKE_ONLY；未启动正式训练；A/B待用户锁定。\n'
for path,text in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',entry),(T/'notes/worklog.md',f'\n- {brt()} BRT E88-W：v3.1工程完成，108 CPU合同，{gpu:.6f}GPUh；多时域/残差/链/阶段/三语言/恢复与profile，最长建议+100；夹爪signed基线修复，正式训练未启动。报告notes/e88_W_v31_20261114.md。\n')]:
 with path.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(text);f.flush();os.fsync(f.fileno())
atomic(ART/'DELIVERY_READY.json',dict(status='COMPLETE_ENGINEERING_SMOKE_ONLY',brt=brt(),report=str(REPORT),report_sha256=sha(REPORT),code_sha256=source_hash(),GPUh=gpu,tests=108,formal_training_started=False,storage_bytes=size,files=[dict(path=str(x),sha256=sha(x)) for x in [run/'contract.json',run/'profile_01/READY.json',run/'gripper_curves.csv',run/'gripper_curves.png',ART/'gpu_ledger.json']]))
print(json.dumps(dict(status='ready',GPUh=gpu,tests=108,formal_estimate=estimate,package_bytes=size)))
