from common import *
import csv,fcntl,hashlib
from e88.audit import HorizonIndex
D=R/'dev';caps=json.loads((D/'captures_full.json').read_text());rows=json.loads((D/'windows_full.json').read_text());ix=HorizonIndex(R/'features',caps);missing=json.loads((R/'missing_parent_audit_resolved.json').read_text());cov=[]
for c in caps:
 ws=[r for r in rows if r['canonical_id']==c['canonical_id']];valid={};bad={}
 for h in H:
  reasons=collections.Counter(ix.resolve(r,h)[1] or 'valid' for r in ws);valid[str(h)]=reasons.pop('valid',0);bad[str(h)]=dict(reasons)
 cov.append(dict(parent=c['parent_component'],canonical_id=c['canonical_id'],demo=ws[0]['stratum'],windows=len(ws),valid=valid,reasons=bad,status='cached',source='published_rev4' if c['canonical_id'] in {x['canonical_id'] for x in json.loads((D/'captures.json').read_text())} else 'recovered_raw'))
for m in missing:
 if m['E93_status'].startswith('recovered'):continue
 family='demo3' if any('demo3' in x for x in m['members']) else ('demo6' if any('demo6' in x for x in m['members']) else 'metadata_only')
 cov.append(dict(parent=m['parent'],canonical_id=None,demo=family,windows=0,valid={str(h):0 for h in H},status=m['E93_status']))
assert len(cov)==51
strata={s:dict(total_parents=sum(r['demo']==s for r in cov),cached_parents=sum(r['demo']==s and r['windows']>0 for r in cov),windows=sum(r['windows'] for r in cov if r['demo']==s),valid={str(h):sum(r['valid'][str(h)] for r in cov if r['demo']==s) for h in H}) for s in sorted({r['demo'] for r in cov})}
atomic(R/'coverage_final.json',dict(parents=cov,strata=strata,frozen_parents=51,cached_parents=27,windows=len(rows),policy='one capture per parent, unchanged old48; up to8 valid fixed-quantile anchors/new parent; no exhaustive-anchor claim'))
protected={}
for folder in ('code_e82','code_e86','code_e88','code_e90'):
 for p in sorted((T/folder).rglob('*.py')):protected[str(p)]=sha(p)
for p in (CK,T/'results/e90_p1_20261114_01/ckpt1_dev_eval.json',T/'results/e90_p1_20261114_01/dev_records.json'):protected[str(p)]=sha(p)
atomic(R/'protected_source_manifest.json',protected)
curve=list(csv.DictReader((T/'results/e90_p1_20261114_01/training_curves.csv').open()));stats=[]
for lo,hi in ((1,500),(2501,3000),(3001,4000),(4001,4500),(4501,5000),(4001,5000)):
 cc=[r for r in curve if lo<=int(r['step'])<=hi];xs=np.array([float(r['step']) for r in cc]);row=dict(first=lo,last=hi)
 for key in ('loss','world'):
  y=np.array([float(r[key]) for r in cc]);row[key+'_mean']=float(y.mean());row[key+'_OLS_per1000']=float(np.polyfit(xs,y,1)[0]*1000)
 stats.append(row)
atomic(R/'curve_diagnostic.json',stats)
lines=['\n## A 缓存与完整冻结名单覆盖（完成可恢复部分）',
'原6parent全部1134相机帧重放：fp16最大绝对差0；新缓存覆盖旧帧最大差0。预处理、RGB/边缘padding/224缩放、三相机顺序、原始帧索引、固定2×4有效面积池化、fp16及本地SigLIP2文件SHA均匹配原训练合同。+100仍从同一2×4缓存按相机平均成global，不另换编码器或归一化。',
'原manifest仅26个dev parent；补查全部51个冻结parent，找回1个原始demo6失败录制，未改变其dev归属。可评27parent/211窗。其余18个仅metadata、5个缺必要时钟合同、1个74帧不满足既有anchor≥24且+50的窗口合同；没有捏造时钟/平移帧或将缺失当静止。故“全部51parent都生成有效缓存”客观未完成。metadata文件已只读核查，索引和原因见missing_parent_audit_resolved/metadata_only_audit.json。',
'窗口规则：旧48窗及所有obs/frame索引原样保留；新parent沿用16个固定分位候选取前8个合法窗、一parent一canonical，不按误差筛选；因此是完整名单审计和全部可恢复parent覆盖，不是对所有可能anchor穷举。历史P3质量标签不用于dev标注。dev语言始终原固定人工短指令。',
'|层|冻结parent|有缓存parent|窗|+10|+25|+50|+100|','|---|---:|---:|---:|---:|---:|---:|---:|']
for s,r in strata.items():lines.append('|'+s+'|'+ '|'.join(map(str,[r['total_parents'],r['cached_parents'],r['windows'],*[r['valid'][str(h)] for h in H]]))+'|')
lines+=['\n全部51parent逐项表：','|parent|层|窗|+10|+25|+50|+100|状态|','|---|---|---:|---:|---:|---:|---:|---|']
for r in sorted(cov,key=lambda z:(z['demo'],z['parent'])):lines.append('|'+r['parent']+'|'+r['demo']+'|'+ '|'.join(map(str,[r['windows'],*[r['valid'][str(h)] for h in H]]))+'|'+r['status']+'|')
lines+=['\n## B1 CPU float32 / 4 Euler / N=1 固定v1结果','原v1共145条有效时域记录：copy/linear最大差0、模型MSE最大差4.768e−7；新增47条原48窗的+100记录不属于原v1有效记录，不能用新旧记录总数相等作复现断言。该收尾断言首次失败已保留；配对修复见cpu_validated_status.json。','下表各parent内平均后parent等权，skill=1−均值MSE之比；95%区间为2000次配对parent bootstrap（seed90）。没有用dev选ckpt、样本数、阈值或模型配置。','|池|+帧|窗/parent|模型MSE|复制MSE|线性外推MSE|复制skill % [95%CI]|残差方向cos|','|---|---:|---|---:|---:|---:|---|---:|']
for name in ('dev','train64'):
 dd=json.loads((R/(name+'_v1_summary.json')).read_text())
 for h,r in dd.items():
  v=r['values'];ci=r['ci95']['skill_copy'];lines.append(f'|{name}|{h}|{r["windows"]}/{r["parents"]}|{v["model"]:.6f}|{v["copy"]:.6f}|{v["linear"]:.6f}|{100*v["skill_copy"]:.3f} [{100*ci[0]:.3f}, {100*ci[1]:.3f}]|{v["cosine"]:.5f}|')
lines+=['train64：seed93从242个训练parent无放回固定抽64，取各parent原训练窗口等距最多8窗，共507窗；与所有dev parent交集0，未补/写训练缓存。+100只有396窗/61parent可用，原因按原缓存/时间合法性保留，详见selection与逐窗记录。训练侧同样固定短指令，排除语言差异。','\n### B 训练曲线斜率（只读E90曲线）','|步段|总loss均值|FM均值|总loss OLS/1000步|FM OLS/1000步|','|---|---:|---:|---:|---:|']
for r in stats:lines.append(f'|{r["first"]}–{r["last"]}|{r["loss_mean"]:.5f}|{r["world_mean"]:.5f}|{r["loss_OLS_per1000"]:.5f}|{r["world_OLS_per1000"]:.5f}|')
lines+=['斜率是噪声训练批次的描述量，没有独立置信区间；最后1000步已包含链权重固定后的区段。不能用首末单步或总loss平台声称世界预测已收敛/已达到不可约下界。']
with (T/'notes/e93_dev_diag_20261114.md').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(strata=strata,curve=stats[-1]),ensure_ascii=False))
