import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json, inspect, hashlib, time, torch, transformers
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer, SiglipImageProcessor
from common import *
torch.set_num_threads(2)
M=pathlib.Path('/mnt/workspace/dengjianfeng/models/siglip2')
out={'timestamp_brt':brt(),'model_id':json.loads((M/'revision.json').read_text())['modelId'],'revision':json.loads((M/'revision.json').read_text())['sha'],'transformers':transformers.__version__,'torch':torch.__version__,'files':[p.name for p in M.iterdir()],'missing_files':['preprocessor_config.json','tokenizer.model'],'downloads':0,'device':'cpu','synthetic_only':True}
with safe_open(M/'model.safetensors',framework='pt',device='cpu') as f:
    keys=list(f.keys()); out['tensor_count']=len(keys)
    wanted=['text_model.embeddings.token_embedding.weight','text_model.embeddings.position_embedding.weight','vision_model.embeddings.patch_embedding.weight','vision_model.embeddings.position_embedding.weight','vision_model.head.probe','text_model.head.weight','vision_model.head.attention.in_proj_weight']
    out['tensor_shapes']={k:list(f.get_slice(k).get_shape()) for k in wanted if k in keys}
    out['parameter_count']=sum(torch.tensor(f.get_slice(k).get_shape()).prod().item() for k in keys)
model,loading=AutoModel.from_pretrained(str(M),local_files_only=True,output_loading_info=True)
out['loading_info']={k:sorted(v) if isinstance(v,set) else v for k,v in loading.items()}; assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading['mismatched_keys'],loading
model.eval(); model.requires_grad_(False)
out['config']=model.config.to_dict()
tok=AutoTokenizer.from_pretrained(str(M),local_files_only=True,use_fast=True)
texts=['This is a photo of a right arm.','This is a photo of a left arm.','画面中，右夹爪向左移动。']
enc=tok([s.lower() for s in texts],padding='max_length',max_length=64,truncation=True,return_tensors='pt')
out['tokenizer']={'class':type(tok).__name__,'is_fast':tok.is_fast,'vocab_size':tok.vocab_size,'token_lengths':(enc['input_ids']!=tok.pad_token_id).sum(-1).tolist(),'padding_id':tok.pad_token_id,'ids_max':int(enc['input_ids'].max()),'returned_keys':list(enc)}
out['default_processor']=SiglipImageProcessor().to_dict()
head=model.vision_model.head
out['pool_head_source']=inspect.getsource(type(head))
pixels=torch.linspace(-1,1,3*224*224).reshape(1,3,224,224)
start=time.monotonic()
with torch.inference_mode():
    v=model.vision_model(pixel_values=pixels)
    t=model.get_text_features(**enc)
    if not isinstance(t,torch.Tensor): t=t.pooler_output
    h=v.last_hidden_state
    singleton=head(h.reshape(-1,1,h.shape[-1])).reshape(h.shape)
    all_one=head(h[:,0:1].expand(-1,196,-1))
    one=head(h[:,0:1])
    sim=torch.nn.functional.normalize(singleton,dim=-1)@torch.nn.functional.normalize(t,dim=-1).T
out['synthetic']={'elapsed_s':time.monotonic()-start,'patch_shape':list(h.shape),'global_shape':list(v.pooler_output.shape),'text_shape':list(t.shape),'map_shape':list(sim.shape),'all_finite':bool(torch.isfinite(sim).all()),'singleton_repeat_max_abs':float((all_one-one).abs().max()),'global_norm':v.pooler_output.norm(dim=-1).tolist(),'map_range':[float(sim.min()),float(sim.max())]}
out['model_sha256']=hashlib.file_digest((M/'model.safetensors').open('rb'),'sha256').hexdigest()
atomic(P/'model_audit.json',out)
atomic(C/'preprocessing_pending.json',{'status':'pending_official_processor_verification','reason':'local preprocessor_config.json missing; no real images processed','default_processor':out['default_processor']})
append_report('## 0 离线权重核验完成\n\n'+brt()+' 型号 '+out['model_id']+'，revision '+out['revision']+'，参数 '+str(out['parameter_count'])+'，双塔权重strict加载无missing/unexpected/mismatch。固定224/16，196patch×768，不是NaFlex。fast tokenizer可离线加载（64长度、显式lowercase）；缺tokenizer.model不妨碍本轮fast接口。缺preprocessor_config.json，预处理仍待核实。CPU合成输入通过，指标并非真实对齐结果。详见 `'+str(P/'model_audit.json')+'`。\n\nMAP头对196个最终视觉token用可学习probe作单查询注意力，再 residual MLP 得到全局向量；probe不随文本变化。主方法逐patch运行同一个MAP头（singleton-MAP），再归一化与冻结文本向量余弦，14×14图逆仿射到原像素；另报原始patch余弦。singleton attention权重恒1，等价于V/out投影+残差MLP，输出维度对齐但训练分布不同；patch已有全图上下文，因此不是精确分割或概率。MAP attention本身不叫文本定位。16像素patch易丢失小尖端/锁孔，色彩/左右对照必须实测。')
update(model_audit_complete=True)
print(json.dumps({k:out[k] for k in ['model_id','parameter_count','tensor_shapes','tokenizer','synthetic']},ensure_ascii=False))
