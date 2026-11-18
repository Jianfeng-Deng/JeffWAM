import sys,os,time,fcntl
sys.dont_write_bytecode=True
sys.path.insert(0,'/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e101a')
from diag import *
for f in ('sampling_metrics.json','predictions.npz','probe_metrics.json','t0_metrics.json','input_ablation_metrics.json'):
 while not (OUT/f).exists():time.sleep(3)
while json.loads((OUT/'status_gpu.json').read_text())['status'] not in ('complete','failed'):time.sleep(3)
assert json.loads((OUT/'status_gpu.json').read_text())['status']=='complete'
status('finalizing')
sampling=json.loads((OUT/'sampling_metrics.json').read_text());probe=json.loads((OUT/'probe_metrics.json').read_text());t0=json.loads((OUT/'t0_metrics.json').read_text());ablation=json.loads((OUT/'input_ablation_metrics.json').read_text());preds=np.load(OUT/'predictions.npz');audit=json.loads((OUT/'contract_audit.json').read_text())
def row(label,m):
 a,b=m['8'],m['50'];return f"|{label}|{a['joint_L1_rad']:.6f}|{a['gripper_L1_mm']:.3f}|{b['joint_L1_rad']:.6f}|{b['gripper_L1_mm']:.3f}|{100*a['joint_skill']:.1f}/{100*a['gripper_skill']:.1f}|{100*b['joint_skill']:.1f}/{100*b['gripper_skill']:.1f}|"
def table(items):return '|条件|前8关节rad|前8夹爪mm|H50关节rad|H50夹爪mm|前8 skill关节/夹爪 %|H50 skill关节/夹爪 %|\n|---|---:|---:|---:|---:|---|---|\n'+'\n'.join(row(k,v) for k,v in items)
report('## 2 采样诊断完整结果\n先窗内聚合，再parent等权。前8与H50均为前缀平均，非端点；L1和MSE skill分开，skill=1−同物理组parent平均MSE/复制MSE（不混合rad与m），不裁负值。S为Euler步数；N均值在**绝对物理动作坐标**平均，种子90..105，同S共用种子、N1/8是N16前缀。无dev选型或阈值调参。\n\n'+table([('复制',audit['dev']['copy']),*sampling['dev'].items(),('t0_zero_velocity_CPU',t0['dev'])])+ '\n\nC2没有训练过的确定性动作均值头。zero是在归一化噪声坐标从0进行Euler，不是零物理动作或数学条件均值。t0_zero_velocity为a=0,t=0读取一次v并反变换；理想FM极限v(0,0,x)=E[y|x]，但本模型t采样logit-normal、精确t=0属边界外推，仅作诊断。t=1输出是速度，不可当绝对动作/均值；未把它误当均值报告。\n完整逐维RMSE/skill见sampling_metrics.json，原始预测见predictions.npz；t0_CPU全部112窗、无CUDA。')
# Per-dimension table, all methods retained in JSON; representative fixed comparisons, not selected best.
labels=['L1','L2','L3','L4','L5','L6','Lgrip','R1','R2','R3','R4','R5','R6','Rgrip'];lines=[]
for j,label in enumerate(labels):
 factor=1000 if j in GRIPS else 1;unit='mm' if j in GRIPS else 'rad';vals=[]
 for meth in ['copy','S4_N1','S4_N8','S16_N16','S16_zero']:
  mm=audit['dev']['copy'] if meth=='copy' else sampling['dev'][meth];vals.extend([mm[h]['dim_L1'][j]*factor for h in ('8','50')])
 lines.append('|'+label+' '+unit+'|'+'|'.join(f'{v:.5f}' for v in vals)+'|')
report('### 逐维 L1（dev48/6parent）\n|维|复制8|复制50|S4N1 8|S4N1 50|S4N8 8|S4N8 50|S16N16 8|S16N16 50|S16zero 8|S16zero 50|\n|---|'+'---:|'*10+'\n'+'\n'.join(lines))
report('## 3 CPU可学习性参照\n训练145parent×16固定等距窗=2320窗，dev48/6；另64个训练parent各中位窗，parent按字典序固定。所有组训练/评估parent不交叉；ep142断言排除。输入state含当前q14、−24/−18/−12/−6因果q差56、当前FK18、历史FK差72、历史valid4，共164维。state_image另加当前24×768冻结SigLIP2 token的固定Gaussian投影128维（seed101、无拟合；保留所有空间token参与，但不是完整token容量上界）。没有使用未来图像、未来状态、parent/window ID作输入。目标两组均为50×14相对signed当前q的增量，包括夹爪。\n岭回归alpha仅在每5个训练parent取1个的内部验证parent上选择，再用全部2320窗重拟合；alpha网格0.1/1/10/100/1000，截距不正则化。小MLP为128 ReLU隐藏层，700输出，零初始化末层，AdamW lr.001、wd.01、batch128、固定1000更新；无dev早停。MLP目标按训练每offset/维std缩放。所有CPU探针CUDA未初始化；具体alpha和损失见probe_config.json。\n\n'+table([(k,v) for k,v in probe.items() if k.endswith('dev')])+ '\n\n### 输入消融（同一CPU岭回归流程）\n'+table([(k,v) for k,v in ablation.items() if k.endswith('dev')])+ '\nC2 action_only保留视觉历史、当前q_normalized14，但**没有因果关节历史/FK条件**：e99w2_model.forward只有w非空时才加入robot_proj(context-derived trajectory)，action_only删除W/R。这属于有意输入布局，不是评估反归一化bug；探针额外历史输入是可学习性参照，不是同输入架构胜负证明。processed14将负夹爪截0，signed14保留负值；两者差很小，无法解释现有夹爪大误差。\n关节按L1：历史岭回归/MLP能胜复制，当前14维岭回归不能；故支持“历史信号可用但现动作路径未利用”，不能宣称只凭当前状态也能线性胜L1复制。按MSE，当前14维岭回归也有小幅正skill，说明MSE/L1结论不同。夹爪前8/H50各探针L1都劣复制，不能把正MSE skill解释为静止段不误动。图像版本没有稳定L1增益，固定投影+小样本不支持否定视觉本身。')
report('### 4.1 同口径训练64窗结果\n'+table([('复制train64',audit['train64']['copy']),*sampling['train64'].items(),('t0_zero_velocity_CPU',t0['train64']),*[(k,v) for k,v in probe.items() if k.endswith('train64')]])+'\n训练64来自拟合所用145parent，其中中位窗不保证与16等距拟合窗逐窗重合，所以是训练parent分布诊断，不是记忆64个ID的过拟合测验。train/dev误差必须结合各自复制误差看，训练copy更容易；两者都没胜复制时，不能仅用原始train小于dev判泛化。')
# Gripper static / event analysis; thresholds specified diagnostics, no dev tuning.
gs={};gates={}
for name in ('train','train64','dev'):
 d=np.load(OUT/(name+'.npz'));q=d['q'][:,GRIPS];y=d['y'][:,:,GRIPS];p=d['parents'];delta=y-q[:,None];steps=np.diff(np.concatenate([q[:,None],y],1),axis=1);avg=lambda z:np.mean([z[p==pp].mean(0) for pp in sorted(set(p))],0)
 stat={}
 for tau in (.0001,.0005,.001):
  st=np.abs(steps)<=tau;cum=np.abs(delta)<=tau;near=np.zeros_like(st)
  events=np.abs(steps)>tau
  for off in range(-5,6):
   if off>=0:near[:,off:] |= events[:,:50-off]
   else:near[:,:off] |= events[:,-off:]
  stat[str(tau*1000)]=dict(per_frame_static_fraction=avg(st.mean(1)).tolist(),anchor_hold_fraction_8=avg(cum[:,:8].mean(1)).tolist(),anchor_hold_fraction_50=avg(cum.mean(1)).tolist(),event_near5_fraction=avg(near.mean(1)).tolist())
  if name in ('dev','train64'):
   gg={}
   for method in ('S4_N1','S16_N16','S16_zero'):
    a=preds[name+'_'+method][:,:,GRIPS];e=np.abs(a-y);ee=e*e
    for h in (8,50):
     mask=~cum[:,:h];error=e[:,:h];nearh=near[:,:h];hold=(np.abs(q[:,None]-y[:,:h]));gated=np.where(mask,error,hold);oracle=np.where(mask,0,hold)
     def ratio_sum(num,den):return (avg(num.mean(1))/np.maximum(avg(den.mean(1)),1e-20)).tolist()
     gg[method+'_h'+str(h)]=dict(model_L1_mm=(avg(error.mean(1))*1000).tolist(),oracle_gate_model_L1_mm=(avg(gated.mean(1))*1000).tolist(),oracle_event_and_value_floor_mm=(avg(oracle.mean(1))*1000).tolist(),static_error_mass_fraction=ratio_sum(error*(~mask),error),near_event_error_mass_fraction=ratio_sum(error*nearh,error),near_event_squared_error_mass_fraction=ratio_sum(ee[:,:h]*nearh,ee[:,:h]))
   gates[name+'_tau'+str(tau*1000)+'mm']=gg
 gs[name]=stat
save('gripper_statistics.json',gs);save('gripper_oracle_gates.json',gates)
rows=[]
for name in ('train','train64','dev'):
 for tau,v in gs[name].items():rows.append(f"|{name}|{tau}|"+'|'.join('/'.join(f'{100*x:.1f}' for x in v[k]) for k in ('per_frame_static_fraction','anchor_hold_fraction_8','anchor_hold_fraction_50'))+'|')
report('## 5 夹爪专项\n静止定义并列报告0.1/0.5/1.0 mm诊断网格，未用dev选择阈值。逐帧静止是相邻反馈差≤阈值；可保持是未来反馈与当前signed q差≤阈值，两者不同。列中均左/右百分比，先parent等权；统计来自实测状态，不是模型标注。\n|集合|阈值mm|逐帧静止%|前8可保持%|H50可保持%|\n|---|---:|---|---|---|\n'+'\n'.join(rows))
lines=[]
for name in ('dev','train64'):
 for h in (8,50):
  for method in ('S4_N1','S16_N16','S16_zero'):
   g=gates[name+'_tau0.5mm'][method+'_h'+str(h)];lines.append(f'|{name} {method}|{h}|'+'|'.join('/'.join(f'{(100*x if "fraction" in key else x):.3f}' for x in g[key]) for key in ('model_L1_mm','oracle_gate_model_L1_mm','oracle_event_and_value_floor_mm','static_error_mass_fraction','near_event_error_mass_fraction'))+'|')
report('### 0.5mm门控上限及误差归因（左/右）\n|条件|前缀|原模型mm|神谕变化门+原幅值mm|神谕门+真幅值残差mm|静止误差占比%|事件±5帧误差占比%|\n|---|---:|---|---|---|---|---|\n'+'\n'.join(lines)+'\n神谕变化门用未来实测判断abs(q_h−q0)>0.5mm；静止则输出当前q，否则原模型幅值。最后一个神谕同时使用真幅值，是不可部署的L1理论下限，不是新模型成绩。事件邻域来自逐帧变化>0.5mm并扩展±5帧；邻域覆盖率见gripper_statistics.json，避免仅凭误差占比忽略覆盖。另0.1/1mm全部见JSON；不把dev数值用作部署门阈值。静止与大幅变化混合适合尝试保持/开合分类+连续幅值残差或单独头；离散化不能单靠此表证明事件时间可预测，还须训练parent留出验证漏动/误动及事件时间误差。')
# Figures: static artifacts, no model/threshold selection.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(10,7))
for hi,h in enumerate(('8','50')):
 for gi,(key,unit) in enumerate((('joint_L1_rad','rad'),('gripper_L1_mm','mm'))):
  ax=axes[hi,gi]
  for n in (1,8,16):ax.plot([4,8,16],[sampling['dev'][f'S{s}_N{n}'][h][key] for s in (4,8,16)],'o-',label=f'N={n}')
  ax.plot([4,8,16],[sampling['dev'][f'S{s}_zero'][h][key] for s in (4,8,16)],'s--',label='zero start');ax.axhline(audit['dev']['copy'][h][key],color='black',ls=':',label='copy');ax.set_title(f'prefix {h}, {unit}');ax.set_xlabel('Euler steps');ax.set_ylabel('parent-equal L1');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT/'sampling_curve.png',dpi=150);plt.close(fig)
bins=json.loads((OUT/'training_bins.json').read_text());fig,ax=plt.subplots(figsize=(8,4))
for mode in ('all','action_only','sync','clean','independent'):
 rr=[r for r in bins if r['mode']==mode];ax.plot([r['step'] for r in rr],[r['action'] for r in rr],label=mode)
ax.set(xlabel='P3 update',ylabel='FM action training loss');ax.legend();fig.tight_layout();fig.savefig(OUT/'training_curve.png',dpi=150);plt.close(fig)
report('## 6 历史动作对照（只读文档和既有数值，无天机推理）\n检索EXPERIMENT_LOG及全部notes，293条copy/动作共现证据归档history_search.json。下表只将动作预测指标与动作保持比较；世界latent/RGB/object胜copy、语言方向分类和动作重排序器胜random都不作动作胜复制证据。表内旧数值是历史结果，不作本轮测试真值/阈值/选点。\n\n|实验|动作目标/归一化、指标时域|动作 vs 复制|训练/推理步数；可比性|\n|---|---|---|---|\n|E01 R0s42/R0s7/R1/R2/R3 天机|54D，10步chunk RMSE rad；上游归一化明细日志未记|0.15428/0.15343/0.22138/0.14739/0.14961，均劣hold0.101|训练/采样步数未记录；仅历史同口径，非Piper|\n|E03 PhaseA image_only/latent_image 天机|10×54命令相对实测q，逐offset/维train mean/std，std floor.01rad；物理chunk RMSE|0.07252/0.07182 <0.101；最佳约.0685，胜|MSE，无Euler；30epoch、最佳约5–6，探索选点|\n|E04 A/B flow 天机|同命令残差/逐offset std；10步物理RMSE|.08145/.07978 <.101，胜|30epoch，Euler10；与E03近似但训练配方不同|\n|E05 A/B候选 天机|同上，10步，温度.5|A single/medoid8/world8=.07321/.07012/.07206；B=.07668/.07573/.07599，均<.101；oracle更好但不可部署|ep30/ep26，Euler10；不是N8均值|\n|E06 C腕视/D关节史 天机|同10步命令残差，chunk RMSE|.07770/.07684 <.101，胜|Euler10；日志最佳点，不是固定P3终点|\n|E08 E-DINO 天机|同10步命令残差|.07409 <.101，胜|24epoch、最佳ep20，Euler10|\n|E12 E候选/进度选择 天机|同上|progress/random/half-noise/medoid16=.07658/.07496/.06988/.06873，均<.101；oracle .05790仅上限|基于E终点，Euler10；选样本≠平均动作|\n|E14 F语义框 天机|同10步命令残差|.07389 <.101，胜|沿E配方，Euler10；与当前Piper不可直接比较|\n|E24 天机|日志称动作门失败，未提供匹配hold数值|不能量化动作相对复制，不把world负结果代替动作数值|25epoch/17475更新；不可直接比较|\n|E32 student4ep 天机|32×54命令残差、train逐维mean/std|日志称全时域劣hold|4epoch；采样Euler10见e32_model；未取本轮新样本|\n|E32无教师/有教师40ep 天机|同32命令残差，FK末端位置单端点8/16/32，mm；手关节L1|无教师+16 L/R45.5/58.0<54.9/70.7；+32 66.5/88.0<81.7/98.1；教师46.3/60.1、69.3/92.1亦胜位置hold；部分旋转/短手仍劣|40epoch，Euler10；266窗9episode，与Piper前缀L1不同|\n|E33 B/B_s2/C/D 天机|同32命令残差，FK单端点；e33_model反变换q+norm*std+mean|+32位置L/R分别55.64/72.33、55.14/74.26、60.68/77.55、59.31/76.63，均低于81.71/98.08；手+32亦低于hold；旋转非全胜|各80epoch，Euler10；逐步全指标归档historical_e33_terminals.json|\n|E35-vis 重评E33-B 天机|区分指令与未来反馈；一步EEF位置/旋转|对命令13.64<14.37mm而旋转2.72>2.34°；对q(t+2)位置12.10>7.41、旋转2.62>1.63，劣复制；+32位置仍胜|ep80 CPU固定seed与旧CUDA不全同；说明目标类型决定胜负|\n|E33sub R1/R2 天机|动作相对B generated1.169/1.164 vs1.160；该值是别的归一化比较|仅证明劣B，缺匹配hold，不能列动作胜/劣copy|60epoch，用户已停|\n|E36b C/T×B/B_s2 天机|32命令残差；chunk joint RMSE与FK；C=hold-conditioned memory，**不是动作保持基线**|C/T=.1163/.1163、.1146/.1150；T无一致收益。EEF+32位置约55–74mm低于原81.7/98.1，但无配对hold CI；q-hold回退列是memory回退而非复制动作|746更新，K16、Euler10；原262有限窗/修复266需区分；只读RESULTS.md|\n|E74 D0 天机|4样本动作均值.47036 vs完整单候选oracle.47473|缺hold，不当作胜copy证据|无新训练；动作池比较|\n|E86-M Piper|50×14：12关节残差+grip绝对signed m、每offset std；前8/H50 L1|关节.010913/.062405 >.010101/.059964；grip15.504/15.286 >旧1.002/3.911|烟雾+100步profile，Euler10；旧夹爪复制有截零bug，模型误差仍远大|\n|E86-V3 Piper|同上|关节.012742/.069426、grip16.066/16.315，均劣旧copy|烟雾，旧grip baseline须改signed|\n|E88独立夹爪MLP Piper|64真实训练parent、当前q+SigLIP2，绝对signed gripper，小头过拟合|前8/H50约.00292/.00348mm，优于训练copy .575/3.222；不是dev泛化|188 CPU更新；E88完整FM仍未胜signed copy；E88 no-action世界成绩不作动作证据|\n|E90/E94/E96/E98 Piper|主要无动作世界头训练/世界评估|未检到匹配动作vs复制结果；不得用世界skill充当动作skill|本轮未补天机/Piper旧ckpt采样|\n|E99-W2 Piper|同现有next-frame反馈、signed复制、前8/H50物理L1|N1关节.016224/.114060、grip4.418/9.590；N8 .012632/.072838、4.074/5.950；均劣copy .010101/.059964、.385/3.336|固定P3 5000，Euler4、N1/8，48/6；与本轮C2最接近可直接比较|\n|E100-C C2 Piper|与E99-W2同合同|本轮重新评估见第2节；原E100报告主要世界，不把世界胜copy当动作胜|固定5000同48/6、Euler4 N1/8严格可比；C1与E99-W2同seed终点模型相等，C2链修正并非纯动作头消融|\n\n关键来源：EXPERIMENT_LOG的E01–E14/E32/E35/E86；tools/train_world_action_net_20261031.py:228–246、train_ffn_cd_20261031.py；notes/e86_M_v3_smoke_20261113.md、e86_V3_integrate_20261113.md、e88_W_v31_20261114.md、e99w2_planned_traj_20261115.md；runs/e33_20261104/*/eval_final/ep80/student_summary.json及B_s2/eval_ep80；runs/e36b_20261104/RESULTS.md。未重新使用模拟/retarget数据。历史某些训练/采样步数未记录即保留未知，不编造。\n天机多数目标是t起点的控制**命令**，当前Piper是q(t+1..t+50)反馈；天机54维全角度、Piper14维混合rad/米；10/32与50步、chunk RMSE/单端点FK与前缀L1、是否先parent等权也不同。因此历史正例只能说明方法并非必然输复制，不能拿数值横向排名当前动作头。')
ledger=json.loads((OUT/'gpu_ledger.json').read_text());assert ledger['status']=='closed' and not ledger['error'] and ledger['GPUh']<1
report('## 7 结论与最小修正方向（未启动训练）\n1. **优先让动作头直接读取因果q历史/FK，并测试确定性增量头。** 现关节已经是相对当前q残差，不能把“再改关节增量”当新修复；真正缺口是action_only没有历史本体条件。最小配对：原C2初始化，加入当前+4段因果q/FK只供A，保留冻结视觉；独立低容量确定性Δq头或均值辅助头，固定2000步/组。预期证据：同固定训练64和dev48上前8/H50关节L1均低于复制，且历史消融的配对增益仍在；同时记录MSE，不用dev选点。按既有C2约.195s/step外推，2组2000步+评估含1.5倍余量约0.4–0.5GPUh；新输入需重新profile准入。\n2. **夹爪单独做保持/变化与signed Δgrip幅值头。** 夹爪当前绝对值回归与大std下FM采样会在静止段误动；CPU连续Δgrip虽MSE可改善，L1仍未胜复制。使用训练parent量化噪声确定变化定义，分类仅决定保持/变化，再回归幅值或少量开度bins，不能将神谕门当可预测事件。先CPU小头固定≤2000更新验证训练parent留出，约0GPUh、≤0.1CPUh；若信号成立，再固定2000步独立头/基线配对，估算≤0.3GPUh（先profile）。预期：静止误动率下降，开合事件漏动/时间误差不恶化，前8夹爪L1低于.385mm，并报告H50。\n3. **最后才增加FM优化或收紧采样方差。** Euler/N曲线见第2节，N均值只消采样方差，不能修偏差；零噪声/t0也不是现成受训均值头。动作loss已占total约70.6%，无需无证据地大幅增权。若保留FM，做固定5000→15000终点的更长训练或确定性均值+小残差方差配对，先验证训练64的物理误差是否持续下降。新增10000步按既有速度含1.5余量约0.82GPUh/组，额外评估另约0.1h；固定终点、训练内定超参。\n\n三类结论须按组/指标讲：关节“线性和小MLP（含历史）都能赢复制L1，C2现路径未稳定实现”；当前14维输入L1线性也赢不了、MSE却能赢，既有口径差异也有历史信息瓶颈；夹爪“线性/小MLP的L1都未胜复制”，但MSE正skill与E88训练过拟合排除了笼统的不可学习结论。训练64与dev均需读完整同口径表；单seed、6dev parent、有限训练抽样不支持真实机器人控制成功或不可预测性定论。\n\n## 资源、命令与交付边界\n'+f'GPU仅已有C2推理；总{ledger["elapsed_s"]:.2f}s = {ledger["GPUh"]:.6f}GPUh，峰值allocated {ledger["peak_allocated_mib"]:.1f}MiB，reserved {ledger["peak_reserved_mib"]:.1f}MiB；准入估算4096MiB、要求空闲≥6144MiB，实际空闲{ledger["start_gpu"][1]}MiB，GPU {ledger["start_gpu"][0]}。2秒监控共享空闲，每次infer检查预算/峰值，未OOM/未杀他人进程，无MPS。失败GPU段0。\n命令：`nohup setsid env PYTHONDONTWRITEBYTECODE=1 .../venvs/e29/bin/python -u code_e101a/diag.py prepare`；随后driver.py分离启动diag.py gpu/probe（各独立状态）；deterministic.py与ablate.py仅CPU；side.py做日志审计；finalize.py做统计/曲线。所有新增仅code_e101a、results/e101a_diag_20261115_01、报告和共享日志追加；旧目录只读import且禁止pyc，未复制模型或大数据，无下载。新增提取为诊断小数组约0.11GiB，最终统计见completion.json。\n已完成七项诊断，无正式训练/新checkpoint。报告只是提出下轮修正，用户若要开展训练需另行指定；本轮没有必须用户决定的阻塞项。')
# Explicit data/resource/source assertions and append-only shared session log.
for p,h in json.loads((OUT/'provenance.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'readonly source changed '+p
for name in ('train','train64','dev'):
 meta=json.loads((OUT/'fixed_windows.json').read_text())[name];assert all(r['cid']!='piper_4dc107335389f5711a0f65e0' for r in meta)
size=sum(p.stat().st_size for root in (OUT,T/'code_e101a') for p in root.rglob('*') if p.is_file());assert size<10*1024**3
text=REPORT.read_text();text=text.replace('状态：进行中。','状态：COMPLETE。').replace('待完成。','已完成，详见后续对应节。');REPORT.write_text(text)
entry=f'\n\n### E101-A — {now()} BRT\n目标：C2 Piper动作头劣于复制的口径/噪声/可学习性/夹爪诊断；无正式训练。改动：新code_e101a只读加载C2，Euler4/8/16×N1/8/16、零起点，CPU岭回归/小MLP与输入消融；固定dev48/6和训练64窗，探针2320窗/145训练parent。命令：nohup setsid .../e29/bin/python code_e101a/diag.py prepare；driver.py；deterministic.py；ablate.py；side.py；finalize.py。目录：results/e101a_diag_20261115_01；报告notes/e101a_action_diag_20261115.md。结果：未发现新反归一化bug；历史状态岭回归关节前8/H50 L1=.00894/.05527rad，优于copy .01010/.05996；夹爪L1仍未胜，正MSE skill不能替代L1。action_only缺少q历史/FK，四模式动作loss系数均1。结论：优先因果本体历史+确定性增量参照、夹爪保持/变化独立头，再考虑更长FM；本轮不训练。GPU={ledger["GPUh"]:.6f}h，新增={size/1024**3:.4f}GiB，状态COMPLETE。\n'
for p,s in [(T/'docs/experiment_log/EXPERIMENT_LOG.md',entry),(T/'notes/worklog.md',f'\n- {now()} BRT E101-A 完成Piper C2动作诊断；关节历史岭回归胜复制，夹爪L1仍劣；无正式训练；GPU {ledger["GPUh"]:.6f}h；notes/e101a_action_diag_20261115.md。\n')]:
 with p.open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(s);f.flush();os.fsync(f.fileno())
save('completion.json',dict(status='complete',brt=now(),GPUh=ledger['GPUh'],new_bytes=size,report=str(REPORT),read_only_hashes_verified=True,no_formal_training=True,no_dev_selection=True));status('complete',new_GiB=size/1024**3)
