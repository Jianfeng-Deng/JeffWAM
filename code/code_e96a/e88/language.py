"""Training-only causal narration embeddings, existing local frozen SigLIP2 CPU."""
import json,numpy as np,torch
from e86.vision import FrozenEncoder,split_sentences
from .audit import read_inputs,labels_index
from .common import *
def build():
    torch.set_num_threads(2);p,ready,ws,stats=read_inputs();labels,source=labels_index(p);dest=ART/'narration_01';dest.mkdir(exist_ok=False)
    allowed={r['canonical_id'] for rs in ws.values() for r in rs};strings=sorted({x['text_en'] for cid,xs in labels.items() if cid in allowed for x in xs if x.get('text_en')});encoder=FrozenEncoder('cpu');chunks=[];index={}
    for s in strings:
        ss=split_sentences(s.lower(),encoder.tokenizer);index[s]=[len(chunks),len(chunks)+len(ss)];chunks.extend(ss)
    arrays=[]
    for i in range(0,len(chunks),32):
        inputs=encoder.tokenizer(chunks[i:i+32],padding='max_length',max_length=64,truncation=False,return_tensors='pt')
        with torch.no_grad():z=encoder.model.get_text_features(**inputs)
        if not torch.is_tensor(z):z=z.pooler_output
        arrays.append(z.float().half().numpy());atomic(dest/'status.json',dict(status='CPU_encoding',done=min(i+32,len(chunks)),total=len(chunks)))
    np.save(dest/'vectors.npy',np.concatenate(arrays));atomic(dest/'index.json',{cid:{str(j):index[x['text_en']] for j,x in enumerate(xs) if x.get('text_en') in index} for cid,xs in labels.items() if cid in allowed})
    atomic(dest/'READY.json',dict(status='ready',source=source,chunks=len(chunks),unique_texts=len(strings),frozen=True,download=False,train_only=True,files=[dict(path=str(dest/n),sha256=sha(dest/n)) for n in ('vectors.npy','index.json')],cuda_initialized=torch.cuda.is_initialized()))
def load(path=ART/'narration_01'):
    path=Path(path);index=json.loads((path/'index.json').read_text());z=np.load(path/'vectors.npy').astype('f4');return {(cid,int(j)):z[sl[0]:sl[1]] for cid,js in index.items() for j,sl in js.items()}
if __name__=='__main__':build()
