import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import sys
sys.dont_write_bytecode=True
import numpy as np,torch,json,pathlib,collections,time
from common import *
from evaluation_data import load_features,motion_dataset,candidate_texts
from encoder import Encoder
from train_heads import fit_standardize,train_classifier,classify
from metrics import normalize,direction_metrics,parent_ci,paired_ci
from contract import sha
def main(version='frozen'):
    root=P/version;root.mkdir(exist_ok=True);features=load_features(version);encoder=Encoder('cpu',str(L/version/'adapter.pt') if version!='frozen' else None)
    results=[];seed=20261112;started=time.monotonic()
    for frame in ['robot']:
        for arm in ['left','right']:
            for horizon in [50,10]:
                stem=f'motion_{frame}_{arm}_h{horizon}';out=root/(stem+'.json')
                if out.exists():results.append(json.loads(out.read_text()));continue
                df,x,start,end=motion_dataset(features,frame,arm,horizon)
                tr=(df.probe_split=='probe-train').to_numpy();va=(df.probe_split=='probe-val').to_numpy()
                if tr.sum()<30 or va.sum()<10:
                    obj={'frame':frame,'arm':arm,'horizon':horizon,'status':'insufficient_samples','train':int(tr.sum()),'val':int(va.sum())};atomic(out,obj);results.append(obj);continue
                y=df.label.to_numpy();truth_angle=np.arctan2(df.dy_m.to_numpy(),df.dx_m.to_numpy());parents=df.parent_component.to_numpy();xt,xv,mean,std=fit_standardize(x[tr],x[va]);rng=np.random.default_rng(seed)
                np.savez_compressed(root/(stem+'_normalization.npz'),mean=mean,std=std)
                # Global image/text embeddings have no native concat temporal projection.
                # Zero-shot concatenation duplicates the frozen sentence [t,t].
                vi=[0] if frame=='screen' else [0,1 if arm=='left' else 2]
                e=normalize(end[:,vi].mean(1));b=normalize(start[:,vi].mean(1));delta=normalize(e-b)
                obj={'frame':frame,'arm':arm,'horizon':horizon,'status':'completed','truth_scope':df.truth_scope.iloc[0],'train_samples':int(tr.sum()),'val_samples':int(va.sum()),'train_parents':len(set(parents[tr])),'val_parents':len(set(parents[va])),'class_train_counts':np.bincount(y[tr],minlength=24).tolist(),'class_val_counts':np.bincount(y[va],minlength=24).tolist(),'epochs':40,'seed':seed,'train_only_standardization':True,'zero_shot':{},'learned':{},'baseline':{},'candidate_template':'F canonical horizontal-direction/amplitude grammar, number/gripper/vertical stripped; explicit frame prefix','pixel_amplitude_train_quantiles':df.attrs.get('pixel_amplitude_train_quantiles')}
                all_text={lang:encoder.text(candidate_texts(arm,frame,lang)) for lang in ['zh','en']}
                for lang,t in all_text.items():
                    dd=delta[va]@t.T;cc=(e[va]@t.T+b[va]@t.T)/2
                    obj['zero_shot'][lang]={'difference':direction_metrics(dd,y[va],parents[va]),'concatenation_duplicated_sentence':direction_metrics(cc,y[va],parents[va]),'candidates':candidate_texts(arm,frame,lang)}
                counts=np.bincount(y[tr],minlength=24);major=np.tile(counts,(int(va.sum()),1));random=rng.random((int(va.sum()),24))
                obj['baseline']={'random':direction_metrics(random,y[va],parents[va]),'majority':direction_metrics(major,y[va],parents[va]),'uniform_theoretical':{'24class_top1':1/24,'24class_top3':3/24,'8direction_top1':1/8,'3amplitude_top1':1/3}}
                prediction_store={'random':random,'majority':major};shuffled=rng.permutation(y[tr])
                for kind in ['linear','contrastive_zh']:
                    text=None if kind=='linear' else all_text['zh']
                    model=train_classifier(xt,y[tr],text,seed=seed,epochs=40);logits=classify(model,xv,text)
                    control=train_classifier(xt,shuffled,text,seed=seed,epochs=40);control_logits=classify(control,xv,text)
                    obj['learned'][kind]={'metrics':direction_metrics(logits,y[va],parents[va]),'shuffled_training_labels':direction_metrics(control_logits,y[va],parents[va]),'top1_gap_vs_shuffled':paired_ci(logits.argmax(-1)==y[va],control_logits.argmax(-1)==y[va],parents[va]),'trainable_parameters':sum(p.numel() for p in model.parameters())}
                    torch.save(model.state_dict(),root/(stem+'_'+kind+'.pt'));prediction_store[kind]=logits;prediction_store[kind+'_shuffled']=control_logits
                all_scores=dict(prediction_store)
                all_scores.update({'zero_difference_'+lang:delta[va]@t.T for lang,t in all_text.items()})
                obj['continuous_angle_error_deg']={}
                for name,sc in all_scores.items():
                    predicted=(sc.argmax(-1)//3)*np.pi/4
                    error=np.abs(np.angle(np.exp(1j*(predicted-truth_angle[va]))))*180/np.pi
                    obj['continuous_angle_error_deg'][name]=parent_ci(error,parents[va])
                np.savez_compressed(root/(stem+'_predictions.npz'),labels=y[va],parents=parents[va].astype(str),feature_index=df.feature_index.to_numpy()[va],zero_difference_zh=delta[va]@all_text['zh'].T,**prediction_store)
                atomic(out,obj);results.append(obj)
                print('MOTION',frame,arm,horizon,'n',int(tr.sum()),int(va.sum()),'elapsed',round(time.monotonic()-started),flush=True)
                append_report('## 运动探针完成 '+version+' '+frame+' '+arm+' h'+str(horizon)+'\n\n'+json.dumps({'train':obj['train_samples'],'val':obj['val_samples'],'val_parents':obj['val_parents'],'truth_scope':obj['truth_scope'],'zero_zh_direction':obj['zero_shot']['zh']['difference']['direction_top1'],'linear_direction':obj['learned']['linear']['metrics']['direction_top1'],'contrast_direction':obj['learned']['contrastive_zh']['metrics']['direction_top1']},ensure_ascii=False)+'\n\n完整top1/top3、角误差、24类混淆矩阵、随机/多数/打乱对照与parent CI见 `'+str(out)+'`。实验2输入为窗口终点当前图与起点历史图，终点等到真实图像receipt后才可观测；并非用起点去预测未来。机器人系只测FK方向/幅度的关联，不能据此声称精确像素运动真值成立。')
    atomic(root/'READY_motion_results.json',{'status':'completed','version':version,'results':[str(root/f'motion_{r["frame"]}_{r["arm"]}_h{r["horizon"]}.json') for r in results],'cpu_only_head_training':True,'gpu_hours':0,'elapsed_s':time.monotonic()-started,'timestamp_brt':brt()})
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'frozen')
