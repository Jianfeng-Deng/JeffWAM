from run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
analysis=json.loads((R/'worst_parent_analysis.json').read_text());colors=['#ffffff']+list(plt.get_cmap('tab10').colors)+['#333333'];cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-1.5,11.5),cmap.N)
plots=[]
for a in analysis:
 parent=a['parent'];z=np.load(R/(parent+'_frozen_sequences.npz'));fig,axs=plt.subplots(4,1,figsize=(16,4.4),sharex=True,constrained_layout=True)
 for ax,(name,y) in zip(axs,[('Reference (model)',z['truth'])]+[(k,z[k]) for k in ['S0','S1','S2']]):
  ax.imshow(y[None,:],aspect='auto',cmap=cmap,norm=norm,extent=[0,len(y),0,1],interpolation='nearest');ax.set_yticks([]);ax.set_ylabel(name,rotation=0,ha='right',va='center')
 axs[-1].set_xlabel('Original source frame (30 fps nominal)');fig.suptitle(parent+' | rank '+str(a['rank'])+' by frozen S2 accuracy')
 fig.legend(handles=[Patch(color=colors[i+1],label=s) for i,s in enumerate(S)]+[Patch(color='#333333',label='BATCH_WAIT')],loc='lower center',bbox_to_anchor=(.53,-.08),ncol=11,frameon=False)
 path=R/(f"worst_{a['rank']}_"+parent+'_legend.png');assert not path.exists();fig.savefig(path,dpi=140,bbox_inches='tight');plt.close(fig);plots.append(str(path))
text=REPORT.read_text()
for a,p in zip(analysis,plots):text=text.replace(a['plot'],p)
text=text.replace('状态：检查输入','状态：完成（已披露缓存覆盖与非人工参考标签限制）').replace('待完成。','已完成，见下方对应追加章节。')
REPORT.write_text(text)
report('## 三例逐parent诊断补充（只读冻结产物）\n- `pc_69f4550c2ae996f44dd6`：S2准确率36.60%、切换recall@30=0。时间线显示前部持续滞后：真值S03误判S02达244帧、S06误判S05达172帧，后部S07误判S10达129帧。更符合阶段分数相邻阶段混淆与前向/驻留累计滞后；仅从本轮分数不能确认视觉可见性是主因。\n- `pc_60067d4b6b3143f5041d`：准确率38.47%、recall@30=0。原标注明确片段首帧已经S06、前序阶段截断，而全部因果控制器固定从S01初始化；真值S07/S06落入S10分别566/299帧。截断初态不匹配与失败/重试语义是明确风险；下一轮可在训练侧研究因果初态估计，不使用dev初态作oracle。\n- `pc_4c5129844edcb591cdf8`：准确率45.09%、recall@30=11.11%。S06误判S05达858帧、误判S04达207帧，S03误判S02达275帧，说明送入阶段相对对准/取自由端持续落后。应优先改善穿入/穿出和相邻阶段的因果监督、区分度及边界定义；P073不可见只说明机械锁紧不可验，不能用它解释所有S06/S07错误。\n\n三例均未新生成标签，以上为时间线、混淆与原标注说明的诊断推断。已补完整阶段颜色图例。')
print('refined three frozen timeline plots')
