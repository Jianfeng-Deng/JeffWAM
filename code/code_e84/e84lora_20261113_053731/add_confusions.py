from common import *
p=C/'figures.py';s=p.read_text();needle="if __name__=='__main__':main()";extra='''def confusion_figure():
 figdir=P/'figures';figdir.mkdir(exist_ok=True);fig,axes=plt.subplots(2,4,figsize=(13,6),constrained_layout=True)
 labels=['F','FL','L','BL','B','BR','R','FR']
 for ri,arm in enumerate(['left','right']):
  frozen=json.loads((OLDP/'frozen'/f'motion_robot_{arm}_h50.json').read_text());lo=json.loads((P/'lora'/f'motion_robot_{arm}_h50.json').read_text());sh=json.loads((P/'shuffled'/f'motion_robot_{arm}_h50.json').read_text())
  cells=[('Frozen zero',frozen['zero_shot']['zh']['difference']),('LoRA zero',lo['zero_shot']['zh']['difference']),('LoRA+contrastive head',lo['learned']['contrastive_zh']['metrics']),('Shuffled-LoRA zero',sh['zero_shot']['zh']['difference'])]
  for ci,(title,m) in enumerate(cells):
   z=np.asarray(m['direction_confusion'],dtype=float);z=z/np.maximum(z.sum(1,keepdims=True),1);ax=axes[ri,ci];im=ax.imshow(z,vmin=0,vmax=1,cmap='Blues');ax.set_title(arm+' '+title,fontsize=9);ax.set_xticks(range(8),labels,fontsize=7);ax.set_yticks(range(8),labels,fontsize=7);ax.set_xlabel('Predicted');ax.set_ylabel('True own-base direction')
 fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='Per-sample row-normalized fraction');fig.savefig(figdir/'h50_direction_confusions.png',dpi=140);plt.close(fig)

''';s=s.replace(needle,extra+"if __name__=='__main__':main();confusion_figure()");p.write_text(s)
p=C/'report.py';s=p.read_text().replace(" s.append('## 资源、交付与续跑", " s.append(f'结果图：`{P/\"figures/primary_comparison.png\"}`；H50方向混淆矩阵图（逐样本行归一，与parent等权CI的权重不同）：`{P/\"figures/h50_direction_confusions.png\"}`；固定G002/G027/G103/G126的右夹爪与橙带热图：`{P/\"figures\"}`。')\n s.append('## 资源、交付与续跑")
p.write_text(s)
print('confusion_plot_added')
