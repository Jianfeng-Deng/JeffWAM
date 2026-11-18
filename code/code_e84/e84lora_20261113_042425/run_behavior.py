import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import sys
sys.dont_write_bytecode=True
import numpy as np,torch,json,time
import pyarrow.parquet as pq
from common import *
from evaluation_data import load_features,candidate_texts,ROBOT_DIRS,AMPLITUDES
from fk_consumption import truth_contract
from encoder import Encoder
from train_heads import fit_standardize,fit_displacement,displacement
from metrics import parent_ci,normalize,counterfactual_metrics,paired_ci,planar_follow
def main(version='frozen'):
    truth_contract();features=load_features(version);root=P/version;root.mkdir(exist_ok=True)
    encoder=Encoder('cpu',str(L/version/'adapter.pt') if version!='frozen' else None);started=time.monotonic();outputs=[]
    for arm in ['left','right']:
        out=root/f'behavior_{arm}.json'
        if out.exists():outputs.append(str(out));continue
        df=pq.read_table(R/'fk/fk_motion_labels.parquet',filters=[('arm','=',arm),('horizon_steps','=',50),('eligible_for_e84','=',True)]).to_pandas()
        df=df.merge(features['meta'].reset_index(names='feature_index'),on=['canonical_id','parent_component','probe_split','cohort','anchor_source_row'],validate='one_to_one')
        # Primary counterfactual is planar; vertical components are still regressed and reported.
        df=df[df.horizontal_direction.isin(ROBOT_DIRS)&df.amplitude.isin(AMPLITUDES)&(df.horizontal_displacement_m>=.005)].copy()
        idx=df.feature_index.to_numpy();tr=(df.probe_split=='probe-train').to_numpy();va=(df.probe_split=='probe-val').to_numpy();parents=df.parent_component.to_numpy()
        if tr.sum()<30 or va.sum()<10:
            atomic(out,{'status':'insufficient_samples','arm':arm,'train':int(tr.sum()),'val':int(va.sum())});continue
        y=df[['dx_m','dy_m','dz_m']].to_numpy(dtype='f4');direction=df.horizontal_direction.map(dict(zip(ROBOT_DIRS,range(8)))).to_numpy();amp=df.amplitude.map(dict(zip(AMPLITUDES,range(3)))).to_numpy();labels=direction*3+amp
        candidates=candidate_texts(arm,'robot','zh');text_vectors=encoder.text(candidates);text=text_vectors[labels]
        xtrain,xval,mean,std=fit_standardize(features['obs'][idx][tr],features['obs'][idx][va]);s=features['state'][idx];smean=s[tr].mean(0);sstd=np.maximum(s[tr].std(0),np.array([.01]*6+[.001]+[.01]*6+[.001],dtype='f4'));s=(s-smean)/sstd
        rng=np.random.default_rng(20261112);shuffle=rng.permutation(int(tr.sum()));conditions={};predictions={};models={}
        for condition in ['language','no_language','shuffled_language_training']:
            tt=text[tr][shuffle] if condition=='shuffled_language_training' else text[tr]
            model=fit_displacement(xtrain,s[tr],tt,y[tr]/.05,use_language=condition!='no_language',seed=20261112,epochs=60)
            pred=displacement(model,xval,s[va],text[va])*.05;predictions[condition]=pred;models[condition]=model
            err=np.linalg.norm(pred-y[va],axis=-1);conditions[condition]={'displacement_L2_error_m':parent_ci(err,parents[va]),'horizontal_true_direction_cosine':parent_ci((normalize(pred[:,:2])*normalize(y[va,:2])).sum(-1),parents[va]),'trainable_parameters':sum(p.numel() for p in model.parameters())}
            torch.save(model.state_dict(),root/f'behavior_{arm}_{condition}.pt')
        shuffled_eval=text[va][rng.permutation(int(va.sum()))]
        pp=displacement(models['language'],xval,s[va],shuffled_eval)*.05
        conditions['language_shuffled_at_eval']={'displacement_L2_error_m':parent_ci(np.linalg.norm(pp-y[va],axis=-1),parents[va])}
        # Same observation, all eight directions. Magnitude remains "medium" in every request.
        dirs=np.stack([np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)],-1)
        cf={};cf_predictions={};pv=parents[va];truth=y[va]
        for condition,model in models.items():
            bank=[]
            for di in range(8):
                req=np.repeat(text_vectors[di*3+1][None],int(va.sum()),axis=0)
                bank.append(displacement(model,xval,s[va],req)*.05)
            bank=np.stack(bank,1);cf_predictions[condition]=bank
            pflat=bank[:,:,:2].reshape(-1,2);requests=np.tile(dirs,(int(va.sum()),1));trues=np.repeat(truth[:,:2],8,axis=0);ps=np.repeat(pv,8)
            result=counterfactual_metrics(pflat,requests,trues,ps,threshold=.001)
            originaldir=direction[va];changed=np.tile(np.arange(8),(int(va.sum()),1))!=originaldir[:,None]
            result['changed_direction_only']=counterfactual_metrics(pflat[changed.ravel()],requests[changed.ravel()],trues[changed.ravel()],ps[changed.ravel()],threshold=.001)
            # Opposite pairs must produce an actual response, not just numerical angle noise.
            response=np.linalg.norm(bank[:,:4,:2]-bank[:,4:,:2],axis=-1)
            result['opposite_request_mean_prediction_separation_m']=parent_ci(response.mean(-1),pv)
            result['note']='primary cosine/follow are own-base XY; request changes only direction, same medium amplitude; zero <1mm is not success'
            cf[condition]=result
        def follows(bank):
            return planar_follow(bank[:,:,:2],dirs[None],.001).ravel()
        obj={'status':'completed','version':version,'arm':arm,'horizon_steps':50,'train':int(tr.sum()),'val':int(va.sum()),'train_parents':len(set(parents[tr])),'val_parents':len(set(pv)),'language_source':'F deterministic horizontal 8x3 category templates from future FK feedback; z/gripper/numeric cm stripped; short controlled phrasing','causal_inputs':'anchor frozen images+four past images+current measured state; no endpoint/future pixels','epochs':60,'output':'own-arm base 3D displacement in meters; planar counterfactual direction metric','conditions':conditions,'counterfactual':cf,'limitation':'controlled language-to-displacement mapping, not future prediction, real robot control, or multimodal instruction-following proof','candidates':candidates,'gap_language_vs_shuffled_follow':paired_ci(follows(cf_predictions['language']),follows(cf_predictions['shuffled_language_training']),np.repeat(pv,8))}
        obj['gap_language_vs_no_language_follow']=paired_ci(follows(cf_predictions['language']),follows(cf_predictions['no_language']),np.repeat(pv,8))
        obj['training_24class_counts']=np.bincount(labels[tr],minlength=24).tolist()
        seen=np.bincount(labels[tr],minlength=24)[np.arange(8)*3+1]>0
        obj['counterfactual_medium_direction_training_supported']=seen.tolist()
        np.savez_compressed(root/f'behavior_{arm}_predictions.npz',parents=pv.astype(str),true_displacement=y[va],feature_index=idx[va],text_class=labels[va],feature_mean=mean,feature_std=std,state_mean=smean,state_std=sstd,**predictions,**{'cf_'+k:v for k,v in cf_predictions.items()})
        atomic(out,obj);outputs.append(str(out));append_report('## 行为反事实完成 '+version+' '+arm+'\n\n'+json.dumps({'train':obj['train'],'val':obj['val'],'val_parents':obj['val_parents'],'with_language':cf['language'],'no_language':cf['no_language'],'shuffled_language_training':cf['shuffled_language_training']},ensure_ascii=False)+'\n\n同观测逐一更换8个方向、幅度固定medium；从未输入未来图像。句子来自FK未来，结果只说明语言能否控制这个小位移预测器，不是未来预测或实机可控性证据。完整条件/误差/输出范数见 `'+str(out)+'`。')
        print('BEHAVIOR',arm,'elapsed',round(time.monotonic()-started),flush=True)
    atomic(root/'READY_behavior_results.json',{'status':'completed','version':version,'files':outputs,'cpu_only':True,'gpu_hours':0,'timestamp_brt':brt(),'elapsed_s':time.monotonic()-started})
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'frozen')
