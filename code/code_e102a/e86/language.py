"""E83 training-only instruction conditioning; FK/E83 exact-anchor alignment targets."""
import json,collections,numpy as np,torch
from pathlib import Path
from .common import *
from .vision import FrozenEncoder,split_sentences
HUMAN_INSTRUCTION='Pick up the zip tie and thread its pointed end through its own locking head to form a loop.'
def build():
    from .data import Package
    import pyarrow.parquet as pq
    dest=C/'artifacts/language_v1';dest.mkdir(exist_ok=False);p=Package();caps={c['canonical_id']:c for c in p.captures};instructions={};ranges={};source_hashes={};align=collections.defaultdict(list)
    for tier in ('A','B'):
        path=R/f'lang_codex/{tier}/lang_{tier}.jsonl';source_hashes[str(path)]=sha(path)
        for r in rows(path):
            if r['split']!='train' or p.split.get(r['parent_id'])!='train':raise ValueError('Evaluation model language prohibited')
            cid=r['canonical_id']
            if cid not in caps or not caps[cid].get('train_ready_p1'):continue
            instructions.setdefault(cid,r['task_instruction_en'])
            span=r.get('source_frame_range_inclusive',r.get('source_frame_range'));ranges[cid]=span
            for k in r.get('keyframe_captions',[]):
                if not isinstance(k,dict):continue
                a=k.get('source_frame_index');text=k.get('caption_en',k.get('text_en'))
                if a is not None and text:align[(cid,int(a))].append(text)
    fkroot=T/'data/e84_piper_align_v1/fk';receipt=json.loads((fkroot/'READY_fk_labels.json').read_text());source_hashes[str(fkroot/'READY_fk_labels.json')]=sha(fkroot/'READY_fk_labels.json')
    for e in receipt['files']:
        assert sha(e['path'])==e['sha256'];source_hashes[e['path']]=e['sha256']
    table=pq.read_table(fkroot/'fk_motion_labels.parquet',columns=['canonical_id','parent_component','anchor_source_row','horizon_steps','text_en','eligible_for_e84'])
    for r in table.to_pylist():
        if not r['eligible_for_e84'] or r['horizon_steps']!=50 or p.split.get(r['parent_component'])!='train':continue
        cid=r['canonical_id']
        if cid in caps and caps[cid].get('train_ready_p1'):align[(cid,int(r['anchor_source_row']))].append(r['text_en'])
    # Align targets are frozen text vectors. Labels never become eval truths.
    strings={HUMAN_INSTRUCTION}
    strings.update(instructions.values());aligntext={k:' '.join(v) for k,v in align.items()};strings.update(aligntext.values())
    encoder=FrozenEncoder('cpu');unique=sorted(strings);segments=[];slices={}
    for text in unique:
        chunks=split_sentences(text.lower(),encoder.tokenizer);slices[text]=(len(segments),len(segments)+len(chunks));segments.extend(chunks)
    vec=[]
    for i in range(0,len(segments),32):
        inp=encoder.tokenizer(segments[i:i+32],padding='max_length',max_length=64,truncation=False,return_tensors='pt')
        with torch.no_grad():z=encoder.model.get_text_features(**inp)
        if not isinstance(z,torch.Tensor):z=z.pooler_output
        vec.append(z.float().half().numpy())
        if i%320==0:atomic(dest/'status.json',dict(status='encoding_text_CPU',chunks_done=i,total_chunks=len(segments),updated_brt=brt()))
    vectors=np.concatenate(vec);np.save(dest/'vectors.npy',vectors)
    index=dict(instructions={cid:dict(slice=slices[text],range=ranges[cid]) for cid,text in instructions.items()},alignment={cid+':'+str(a):slices[text] for (cid,a),text in aligntext.items()},human_instruction=slices[HUMAN_INSTRUCTION])
    atomic(dest/'index.json',index);atomic(dest/'READY_language.json',dict(status='ready',instructions=len(instructions),exact_anchor_alignment=len(aligntext),unique_texts=len(unique),chunks=len(segments),source_hashes=source_hashes,files=[dict(path=str(dest/n),sha256=sha(dest/n)) for n in ('vectors.npy','index.json')],cuda_initialized=torch.cuda.is_initialized(),forbidden_uses=['eval_truth','thresholds','checkpoint_selection'],no_nearest_anchor=True));print(json.dumps(json.loads((dest/'READY_language.json').read_text())|{'source_hashes':'omitted','files':'omitted'}),flush=True)
def load(windows,path):
    path=Path(path);receipt=json.loads((path/'READY_language.json').read_text())
    for e in receipt['files']:assert sha(e['path'])==e['sha256']
    index=json.loads((path/'index.json').read_text());vec=np.load(path/'vectors.npy').astype('f4');result={};sl=lambda x:vec[x[0]:x[1]]
    result['__human_default__']=sl(index['human_instruction'])
    for r in windows:
        if r['split']!='train':continue
        cid=r['canonical_id'];a=r['anchor'];e=index['instructions'].get(cid)
        if e:
            lo,hi=e['range']
            if lo<=a<=hi:result[(cid,a)]=sl(e['slice'])
        target=index['alignment'].get(cid+':'+str(a))
        if target:result[('align',cid,a)]=sl(target).mean(0)
        elif e and lo<=a<=hi:result[('align',cid,a)]=sl(e['slice']).mean(0)
    return result
if __name__=='__main__':
    torch.set_num_threads(2);build()
