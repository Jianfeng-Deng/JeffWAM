from common import *
p=C/'report.py';s=p.read_text();needle=" rows=[]\n for arm in ['left','right']:\n  for h in [50,10]:"
insert=""" def image_auc(root):
  return next(r['auroc'] for r in load(root/'pixel_results.json')['summaries'] if r['split']=='probe-val' and r['quality']=='main' and r['radius_multiplier']==1 and r['condition']=='correct' and r['word']=='orange zip tie')
 def pair_motion(root,head=None):
  ms=[]
  for arm in ['left','right']:
   a=load(root/f'motion_robot_{arm}_h50.json');m=a['zero_shot']['zh']['difference'] if head is None else a['learned'][head]['metrics'];ms.append(fmt(m['direction_top1'],True))
  return ' / '.join(ms)
 def pair_behavior(root):return ' / '.join(fmt(load(root/f'behavior_{arm}.json')['counterfactual']['language']['follow_rate'],True) for arm in ['left','right'])
 s.append(table(['对齐','冻结零样本','冻结特征小头','正式LoRA'],[
 ['语言↔图像','橙带AUROC '+fmt(image_auc(OLDP/'frozen'),False,3),'未训练像素头','橙带AUROC '+fmt(image_auc(P/'lora'),False,3)+'；左右仍不可判定'],
 ['语言↔机器人运动（H50方向%,左/右）',pair_motion(OLDP/'frozen'),pair_motion(OLDP/'frozen','contrastive_zh'),'原生 '+pair_motion(P/'lora')+'；同结构头 '+pair_motion(P/'lora','contrastive_zh')],
 ['语言↔动作（反事实跟随%,左/右）','编码器无动作输出',pair_behavior(OLDP/'frozen'),pair_behavior(P/'lora')],
 ['语言↔画面像素运动','无真值，未评估','无真值，未评估','无真值，未评估']]))
"""
assert needle in s;s=s.replace(needle,insert+needle,1)
s=s.replace('原逐句parent映射和SHA仍在旧run，不重做审计。','scene_description_en另有320条/320唯一/243parent；FK覆盖243parent。原逐句parent映射和SHA仍在旧run e83_language_information_afc37bf06a7070ab.json与e84p_20261112_153338/fk_information.json，不重做审计；本run11条caption的逐句parent/锚点对应另存training_distribution.json和e83_training_pair_provenance.json。')
p.write_text(s);print('summary_matrix_added')
