import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import collections,math,re,json,pathlib
from transformers import AutoTokenizer
from common import *
from contract import contracts,sha
def stats(pairs):
    d=collections.defaultdict(set);parent_text=collections.defaultdict(set)
    for text,parent in pairs:
        text=' '.join(text.strip().split());d[text].add(parent);parent_text[parent].add(text)
    counts=collections.Counter(t for t,p in pairs);n=sum(counts.values())
    entropy=-sum((v/n)*math.log2(v/n) for v in counts.values()) if n else 0
    return {'texts':n,'unique_sentences':len(d),'parents':len(parent_text),'sentence_entropy_bits':entropy,'max_sentence_parent_count':max(map(len,d.values()),default=0),'mean_parents_per_sentence':sum(map(len,d.values()))/max(len(d),1),'mean_sentences_per_parent':sum(map(len,parent_text.values()))/max(len(parent_text),1),'text_to_parents':{k:sorted(v) for k,v in sorted(d.items())}}
def main():
    split,_,_=contracts();tok=AutoTokenizer.from_pretrained('/mnt/workspace/dengjianfeng/models/siglip2',local_files_only=True,use_fast=True)
    rows=[];excluded=[];sources={};annotation_quality=[]
    for tier in ['A','B']:
        path=T/f'data/e82_piper_v1/lang_codex/{tier}/lang_{tier}.jsonl';payload=path.read_bytes();digest=__import__('hashlib').sha256(payload).hexdigest()
        snapshot=P/f'e83_{tier}_snapshot_{digest[:16]}.jsonl'
        if not snapshot.exists():snapshot.write_bytes(payload)
        sources[tier]={'path':str(path),'sha256':digest,'snapshot':str(snapshot),'snapshot_bytes':len(payload)}
        for line in payload.decode().splitlines():
            r=json.loads(line);parent=r['parent_id'];assert r['split']=='train'
            if parent not in split['parents']:excluded.append(r['annotation_id']);continue
            assert 'test_ground_truth' in r['forbidden_uses'];assert 'language_supervision_training' in r['intended_uses']
            vr=r.get('visual_review',{});annotation_quality.append({'annotation_id':r['annotation_id'],'tier':tier,'parent':parent,'missing_views':r.get('missing_views',[]),'full_episode_hand_sequence_verified':vr.get('full_episode_hand_sequence_verified'),'unobserved_completion_steps_are_prescriptive':vr.get('unobserved_completion_steps_are_prescriptive'),'confidence':r.get('confidence')})
            def add(field,text):
                if not text:return
                ids=tok(text.lower(),truncation=False)['input_ids']
                rows.append({'tier':tier,'parent':parent,'probe_split':split['parents'][parent],'annotation_id':r['annotation_id'],'field':field,'text':text,'tokens':len(ids),'over_64_tokens':len(ids)>64})
            for f in ['scene_description_en','scene_description_zh','task_instruction_en','task_instruction_zh','retrospective_summary_en','retrospective_summary_zh']:add(f,r.get(f))
            for lang in ['en','zh']:
                v=r.get('instruction_variants_'+lang,[])
                if isinstance(v,list):
                    for i,text in enumerate(v):add(f'instruction_variant_{i+1}_{lang}',text if isinstance(text,str) else text.get('text_'+lang,''))
            for p in r.get('process_narration',[]):
                for lang in ['en','zh']:add('process_narration_'+lang,p.get('text_'+lang))
            for p in r.get('keyframe_captions',[]):
                for lang in ['en','zh']:add('keyframe_caption_'+lang,p.get('caption_'+lang))
    results={}
    for tier in ['A','B','all']:
        for field in sorted(set(r['field'] for r in rows)):
            group=[r for r in rows if (tier=='all' or r['tier']==tier) and r['field']==field]
            if not group:continue
            result=stats([(r['text'],r['parent']) for r in group]);result.update(mean_tokens=sum(r['tokens'] for r in group)/len(group),max_tokens=max(r['tokens'] for r in group),truncated_at_64_count=sum(r['over_64_tokens'] for r in group),probe_split_counts=dict(collections.Counter(r['probe_split'] for r in group)))
            results[tier+'/'+field]=result
    aparents={r['parent'] for r in rows if r['tier']=='A'};bparents={r['parent'] for r in rows if r['tier']=='B'}
    overlap={p:sorted({r['annotation_id'] for r in rows if r['parent']==p}) for p in sorted(aparents&bparents)}
    obj={'timestamp_brt':brt(),'sources':sources,'scope':'only E84 allowed parents; model-generated E83 training labels, never test truth','excluded_annotations_outside_E84_parents':excluded,'records':len(set(r['annotation_id'] for r in rows)),'unique_parents':len(set(r['parent'] for r in rows)),'A_B_parent_overlap':overlap,'annotation_quality':annotation_quality,'B_unverified_full_sequence_ids':[r['annotation_id'] for r in annotation_quality if r['tier']=='B' and r['full_episode_hand_sequence_verified'] is False],'fields':results,'rows':rows}
    version=__import__('hashlib').sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:16]
    out=P/f'e83_language_information_{version}.json'
    if not out.exists():atomic(out,obj)
    table='|范围/字段|记录/唯一句子|parent|文本熵bit|>64token|\n|---|---:|---:|---:|---:|\n'
    for key in ['A/task_instruction_en','B/task_instruction_en','B/instruction_variant_1_en','all/task_instruction_en','all/scene_description_en','A/process_narration_en']:
        if key in results:
            r=results[key];table+=f'|{key}|{r["texts"]}/{r["unique_sentences"]}|{r["parents"]}|{r["sentence_entropy_bits"]:.3f}|{r["truncated_at_64_count"]}|\n'
    append_report('## E83信息量实算（固定主文件快照 '+version+'）\n\n'+table+'\n完整text↔parent映射与各split计数存于 `'+str(out)+'`。E83-A仍可能续写，之前103/104等统计是对应时刻快照，最终以本次sha版本为准；不改源文件。精确文本熵不是行为信息量；长文本在64token文本塔下会截断，不能直接假设E83全文信息都被编码。scene/事后叙述不作测试真值；未来过程描述不能作为部署当前观测输入。')
    append_report('A/B主文件parent交集（不改原文件）：'+json.dumps(overlap,ensure_ascii=False)+'。103条记录对应102parent；训练应按parent去重/等权，避免跨tier重复加权。')
    print(json.dumps({'records':obj['records'],'unique_parents':obj['unique_parents'],'excluded':excluded,'selected_fields':{k:{a:v for a,v in r.items() if a!='text_to_parents'} for k,r in results.items() if k in ['A/task_instruction_en','B/task_instruction_en','all/task_instruction_en']}},ensure_ascii=False))
if __name__=='__main__':main()
