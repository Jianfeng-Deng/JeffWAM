from common import *
def f(m,scale=1):return [round(m['estimate']*scale,5),[round(x*scale,5) for x in m['ci95']]]
j=json.loads((P/'paired_comparisons.json').read_text())
for arm in ['left','right']:
 print('MOTION',arm)
 for pair,z in j['motion'][arm+'_h50'].items():
  print(pair,{k:f(z[k],1 if k=='angle_error_reduction_deg' else 100) for k in ['direction_top1','top1','top3','angle_error_reduction_deg']})
 print('BEHAVIOR',arm,{pair:{k:f(m,100 if k=='follow_rate' else 1) for k,m in z.items()} for pair,z in j['behavior'][arm].items()})
 m=json.loads((P/'matched_frozen'/f'motion_robot_{arm}_h50.json').read_text());print('MATCHED',arm,{k:f(m['metrics']['true'][k],100) for k in ['top1','direction_top1']})
for word,a in j['pixel'].items():print('PIXEL_R1',word,{pair:{k:f(m,100 if k=='pointing' else 1) for k,m in z.items()} for pair,z in a['1'].items()})
for word,a in j['semantic_contrast_gain'].items():print('SEMANTIC_GAIN',word,{k:{kk:f(mm,100 if kk=='pointing' else 1) for kk,mm in v.items()} for k,v in a.items()})
d=json.loads((P/'drift_results.json').read_text())
for v,a in d['results'].items():print('DRIFT',v,'val',a['visual']['probe-val'],'textcos',a['text_cosine'],'textCKA',a['text_linear_CKA'])
