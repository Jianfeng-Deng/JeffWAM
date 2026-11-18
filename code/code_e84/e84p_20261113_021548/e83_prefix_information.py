import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,collections,hashlib
from transformers import AutoTokenizer
from common import *
from language_audit import stats
f=sorted(P.glob('e83_language_information_*.json'))[-1];source=json.loads(f.read_text());tok=AutoTokenizer.from_pretrained('/mnt/workspace/dengjianfeng/models/siglip2',local_files_only=True,use_fast=True)
fields=collections.defaultdict(list)
for row in source['rows']:
    if row['field'] not in ['task_instruction_en','task_instruction_zh','instruction_variant_1_en','scene_description_en']:continue
    ids=tok(row['text'].lower(),padding='max_length',max_length=64,truncation=True)['input_ids'];key=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    for tier in [row['tier'],'all']:fields[tier+'/'+row['field']].append((key,row['parent']))
result={'timestamp_brt':brt(),'source_snapshot':str(f),'scope':'same accepted E83 snapshot; no new source annotations; exact64token input sequence rather than complete prose','fields':{k:stats(v) for k,v in fields.items()}}
atomic(P/'e83_64token_information.json',result)
print(json.dumps({k:{a:v for a,v in r.items() if a!='text_to_parents'} for k,r in result['fields'].items() if k.endswith('task_instruction_en')},ensure_ascii=False))
