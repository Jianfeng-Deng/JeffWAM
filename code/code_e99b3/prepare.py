from common import *
import numpy as np,collections
spec=json.loads((T/'data/e97b_stages_v1/stages_predicates_v1.json').read_text()); put('spec.json',spec)
tr=jl(T/'data/e98_predicate_labels_train/predicates.jsonl');dv=jl(T/'data/e98_predicate_labels_dev/predicates.jsonl')
parents=sorted({r['parent_id'] for r in tr});assert len(parents)==60
val=set(np.random.default_rng(9903).choice(parents,12,replace=False).tolist());assert not set(parents)&{r['parent_id'] for r in dv}
put('split.json',dict(seed=9903,fit=[p for p in parents if p not in val],validation=sorted(val)))
sources=[]
sel=json.loads((T/'data/e98_predicate_labels_train/selection.json').read_text())['selected']
for s in sel:
 sources.append(dict(parent=s['parent_id'],cid=s['canonical_id'],pool='train',validation=s['parent_id'] in val,source=resolve(s['source_path']),videos={c:resolve(s['videos'][c]) for c in CAMS},n=s['source_frame_range_inclusive'][1]+1,stratum=s['stratum']))
for s in json.loads((T/'data/e98_predicate_labels_dev/sources.json').read_text()):
 sources.append(dict(parent=s['parent_id'],cid=s['canonical'],pool='dev',validation=False,source=s['source_file'],videos={c:s['cameras'][c]['source_file'] for c in CAMS},n=s['source_frame_range'][1]+1))
assert len(sources)==80 and all(s['cid']!=EXCLUDE for s in sources)
for s in sources:
 assert Path(s['source']).exists() and all(Path(p).exists() for p in s['videos'].values())
 rows=tr if s['pool']=='train' else dv; relevant=[r for r in rows if r['parent_id']==s['parent']]
 frames=sorted({r['source_frame_index'] if s['pool']=='train' else r['frame'] for r in relevant})
 # Full source-frame causal B4 on internal validation and dev. Other train parents only labeled samples + O history.
 targets=list(range(s['n'])) if s['validation'] or s['pool']=='dev' else frames
 ids=sorted({f+off for f in targets for off in (-24,-18,-12,-6,0) if f+off>=0})
 s.update(targets=targets,feature_rows=ids,annotation_frames=frames)
put('sources.json',sources)
for pool,rows in [('train',tr),('dev',dv)]:
 compact=[]
 for r in rows:
  f=r.get('source_frame_index',r.get('frame'));ob=r.get('visibility',r.get('observability'))
  raw=r.get('causal_raw_record') or {}
  compact.append(dict(parent=r['parent_id'],frame=f,predicate=r['predicate_id'],stage=r['stage_id'],answer=r['answer'],visibility=ob,semantics=r.get('answer_semantics','retrospective_event'),causal_valid=bool(r.get('causal_valid_for_b3',False)),causal_answer=r.get('causal_observable_answer'),causal_visibility=raw.get('observability',ob),causal_stage=r.get('causal_stage_id',r['stage_id'])))
 put(pool+'_labels.json',compact)
put('stages_train.json',jl(T/'data/e98_predicate_labels_train/stages.jsonl'));put('stages_dev.json',jl(T/'data/e98_predicate_labels_dev/stages.jsonl'))
protocol=dict(seed=9903,prompt_count=1,prompt='Three synchronized views at the current source frame: overhead, left wrist, right wrist. Answer the following question about the current zip tie using only these images. Output exactly yes or no. If uncertain, give your best binary judgment. Question: {question_en}',vlm_history=0,vlm_max_pixels=448*448,vlm_score='softmax of first-token yes/no logits; lowercase single token; prefix KV reuse',vision='E86 exact aspect padding/valid-mask 224 SigLIP2 2x4 fp16',observation_offsets=[-24,-18,-12,-6,0],observation_meta='nominal source-frame age; synchronized source row indices; no future row',o_readout='A fixed5000 checkpoint, O-only masked subgraph through six blocks; camera means of current O plus valid history mean, 1024D; fixed human default text',proprio='q with raw grip; FK link6 + .1358m tool offset and 6D rotations; causal state_source_brackets right endpoint <= timestamp[t], row<=t; t,t-.5,t-1sec',train_labels='retrospective yes/no; unknown and NA excluded binary; train unknown separate observability classifier',head='b predicate logistic regression; c/d shared 128-SiLU MLP ->21 independent masked BCE logits',candidates=dict(b_C=[.01,.1,1.],mlp_epochs=[10,30],lr=.001,weight_decay=.001),selection='macro balanced accuracy on 12 internal-validation parents; per-predicate threshold grid .2,.35,.5,.65,.8 from internal validation; refit all60 fixed selected epochs; no dev changes',abstention='separate shared unknown-vs-known logistic score, threshold .5 fixed before dev; NA excluded; coverage and unknown recall/precision separately; P073 always abstain if no train yes/no',b4='d fixed; recursive exact rules, unknown never no; initialized S01 without dev labels; update every source frame, at most one transition per frame; P071 prior event, clear on failure/new-cycle/retry; continue enabled stop false; same-priority conflicts hold; threshold candidate persistence 1/3/6 frames selected on validation stage edit distance',bootstrap='2000 parent cluster resamples seed9903; row-weighted within repeated parent clusters, macro across estimable predicates; no threshold/selection from dev',dev_contract='single locked inference artifact; causal main 4074; retrospective supplement uses answer_semantics retrospective_event only; all-key mixed answer appendix separately',limits=dict(gpu_seconds=10800,new_disk_bytes=20_000_000_000))
put('PROTOCOL_LOCK.json',protocol)
hashes={str(p):sha(p) for p in [T/'data/e97b_stages_v1/stages_predicates_v1.json',T/'data/e98_predicate_labels_dev/predicates.jsonl',T/'data/e98_predicate_labels_dev/stages.jsonl',T/'data/e98_predicate_labels_train/predicates.jsonl',T/'data/e98_predicate_labels_train/stages.jsonl',T/'results/e98w_a_20261114_04/ckpt_05000.pt']}
put('input_hashes.json',hashes)
summary=dict(train_parents=60,fit_parents=48,validation_parents=12,dev_parents=20,train_labels=len(tr),dev_labels=len(dv),causal_labels=sum(r.get('causal_valid_for_b3',False) for r in dv),feature_source_rows=sum(len(s['feature_rows']) for s in sources),full_validation_frames=sum(s['n'] for s in sources if s['validation']),full_dev_frames=sum(s['n'] for s in sources if s['pool']=='dev'),feature_bytes=sum(len(s['feature_rows']) for s in sources)*3*8*768*2)
put('data_audit.json',summary);status('prepared',**summary);report('## 数据审计与事前锁定 '+now()+'\n'+json.dumps(summary,ensure_ascii=False)+'\n配置、分割、输入哈希已存 PROTOCOL_LOCK.json / split.json / input_hashes.json。只用一个 VLM 提示模板，不进行提示词选择。train 48/12 parent 内部验证；不读取 dev 指标选型。')
