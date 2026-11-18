from common import *
p=C/'analyze.py';s=p.read_text();needle=" results['semantic_contrast_gain']=semantic\n atomic(P/'paired_comparisons.json',results)";insert=""" results['semantic_contrast_gain']=semantic
 semantic_shuffle={}
 for word,controls in semantic.items():
  rr=[r for r in pixels['frozen'].values() if r['word']==word and r['radius_multiplier']==1 and r['condition']=='correct'];ps=[r['parent'] for r in rr];ks=[tuple(r[k] for k in keys) for r in rr];out={}
  for control in controls:
   ck=[tuple(control if name=='condition' else value for name,value in zip(keys,k)) for k in ks]
   out[control]={metric:paired_ci([pixels['lora'][k][metric]-pixels['lora'][c][metric] for k,c in zip(ks,ck)],[pixels['shuffled'][k][metric]-pixels['shuffled'][c][metric] for k,c in zip(ks,ck)],ps) for metric in ['pointing','auroc']}
  semantic_shuffle[word]=out
 results['semantic_contrast_true_minus_shuffled']=semantic_shuffle
 atomic(P/'paired_comparisons.json',results)""";assert needle in s;s=s.replace(needle,insert);p.write_text(s)
p=C/'report.py';s=p.read_text();needle=" s.append('## 原编码偏离与遗忘";insert=" rows=[]\n for word,a in j['semantic_contrast_true_minus_shuffled'].items():\n  for control,z in a.items():rows.append([word,control,fmt(z['pointing'],True),fmt(z['auroc'],False,4)])\n s.append('### 语义区分差：真实文本LoRA−打乱文本LoRA\\n\\n两组共享恒定橙色HSV正例，以下只反映FK/E83标签正确性引起的额外差异。\\n'+table(['查询','对照','pointing差 pp','AUROC差'],rows))\n";assert needle in s;s=s.replace(needle,insert+needle);p.write_text(s)
p=P/'visual_review.json';j=json.loads(p.read_text());from contract import sha
j['statistical_figures_reviewed_brt']=brt();j['statistical_figures']=[dict(path=str(P/'figures'/n),sha256=sha(P/'figures'/n)) for n in ['primary_comparison.png','h50_direction_confusions.png']];j['statistical_figure_observations']=['Primary plot agrees with result JSON: frozen/true/shuffled zero-shot, same-head directions and counterfactual follow largely overlap; dashed12.5% baseline is labeled.','H50 row-normalized confusion matrices show forward bias in zero-shot and a stronger diagonal with learned head; axes are own-base directions, not pixel directions.'];atomic(p,j)
print('color_shuffle_contrast_and_final_plot_review_added')
