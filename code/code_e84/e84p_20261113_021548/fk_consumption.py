import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import json,re,collections,pathlib,sys
sys.dont_write_bytecode=True
import numpy as np,pyarrow.parquet as pq
from common import *
from contract import contracts,sha
from language_audit import stats
def truth_contract():
    state=json.loads((P/'progress.json').read_text());assert state.get('fk_gt_ready'),'wait watcher 10min gate before opening FK/GT'
    path=R/'READY_fk_gt.json';assert sha(path)==state['fk_gt_ready']['sha256'],'READY changed; stop and audit version'
    ready=json.loads(path.read_text());assert ready['dev_test_data_read'] is False
    split,_,_=contracts();assert ready['probe_split']['sha256']==sha(R/'split/probe_split.json')
    for r in ready['files']:assert sha(r['path'])==r['sha256'],r['path']
    return ready,split
def audit():
    ready,split=truth_contract()
    cols=['label_id','canonical_id','parent_component','probe_split','cohort','anchor_source_row','end_source_row','horizon_steps','arm','coordinate_frame','dx_m','dy_m','dz_m','displacement_m','horizontal_displacement_m','horizontal_direction','vertical_direction','amplitude','gripper','text_en','text_zh','top_projection_eligible','eligible_for_e84']
    fk=pq.read_table(R/'fk/fk_motion_labels.parquet',columns=cols,filters=[('eligible_for_e84','=',True)]).to_pandas()
    assert fk.label_id.nunique()==len(fk);assert set(fk.parent_component).issubset(split['parents'])
    assert (fk.probe_split==fk.parent_component.map(split['parents'])).all()
    # Text uniqueness includes numeric cm; report a number-stripped version as well.
    fields={}
    for which in ['all','probe-train','probe-val']:
        d=fk if which=='all' else fk[fk.probe_split==which]
        for name,values in [('text_en',d.text_en),('text_en_without_numeric',d.text_en.str.replace(r'\d+(?:\.\d+)?','<n>',regex=True))]:
            pairs=list(zip(values.tolist(),d.parent_component.tolist()));fields[which+'/'+name]=stats(pairs)
    obj={'timestamp_brt':brt(),'ready_sha256':sha(R/'READY_fk_gt.json'),'rows':len(fk),'parents':fk.parent_component.nunique(),'fields':fields,'distributions':{c:fk.groupby(['probe_split',c],dropna=False).size().to_dict() for c in []},'limits':ready['known_limits'],'consumption':ready['consumption'],'visual_review':ready['visual_review']}
    atomic(P/'fk_information.json',obj)
    table='|文本范围|句子条数|精确唯一句|parent|熵bit|\n|---|---:|---:|---:|---:|\n'
    for k,r in fields.items():table+=f'|{k}|{r["texts"]}|{r["unique_sentences"]}|{r["parents"]}|{r["sentence_entropy_bits"]:.3f}|\n'
    append_report('## FK/GT接收与适用范围\n\n'+brt()+' 已核验READY所有入口SHA、probe split同版本及train归属。\n\n'+table+'\nFK精确句子的数字厘米值会增加唯一数，去数字版更接近离散运动语义；全映射存fk_information.json。\n\nF交付限制：'+json.dumps(ready['consumption'],ensure_ascii=False)+'\n\n未认证夹爪/整臂/locking head/tip实例精确像素GT；相关测试结果必须标记为coarse/proxy或N/A，不用它们断言SigLIP2失败。不把未标定demo3/demo6投影当像素真值。')
    update(fk_gt_audit_complete=True,status='fk_gt_ready')
    print(json.dumps({'rows':len(fk),'parents':int(fk.parent_component.nunique()),'limits':ready['consumption'],'review':ready['visual_review']},ensure_ascii=False))
if __name__=='__main__':audit()
