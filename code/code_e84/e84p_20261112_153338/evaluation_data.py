import numpy as np,pandas as pd,pathlib,json,collections
from common import *
from contract import contracts,sha
from fk_consumption import truth_contract
from heads import observations
ROBOT_DIRS=['forward','forward-left','left','backward-left','backward','backward-right','right','forward-right']
SCREEN_DIRS=['right','up-right','up','up-left','left','down-left','down','down-right']
ROBOT_ZH=['前','左前','左','左后','后','右后','右','右前']
SCREEN_ZH=['右','右上','上','左上','左','左下','下','右下']
AMPLITUDES=['small','medium','large']
def candidate_texts(arm,frame,language='zh'):
    ds=ROBOT_DIRS if frame=='robot' else SCREEN_DIRS
    zs=ROBOT_ZH if frame=='robot' else SCREEN_ZH
    if language=='en':
        context='In the robot coordinate frame, ' if frame=='robot' else 'In the image, '
        return [context+f'the {arm} arm moves {d} ({a} displacement).' for d in ds for a in AMPLITUDES]
    context='机器人系中，' if frame=='robot' else '画面中，'
    return [context+('左' if arm=='left' else '右')+'臂向'+d+'移动（'+a+'幅）。' for d in zs for a in ['小','中','大']]
def load_features(version='frozen'):
    root=R/'features'/RUN/version;ready=json.loads((root/'READY_features.json').read_text())
    assert sha(ready['selection_path'])==ready['selection_sha256']
    selection=json.loads(pathlib.Path(ready['selection_path']).read_text());groups=collections.defaultdict(list)
    for r in selection['samples']:groups[r['canonical_id']].append(r)
    meta=[];past=[];now=[];endpoints=[];endpoint_valid=[];state=[];obs=[]
    for cid,rows in sorted(groups.items()):
        f=np.load(root/(cid+'.npz'));anchors=f['anchor_source_row'];assert np.array_equal(anchors,[r['anchor_source_row'] for r in rows])
        g=f['global_features'].astype('f4');valid=f['history_valid']
        obs.append(observations(g,valid));now.append(g[:,-1]);endpoints.append(f['endpoint_global_features'].astype('f4'));endpoint_valid.append(f['endpoint_valid']);state.append(f['state'])
        for i,r in enumerate(rows):meta.append({k:r[k] for k in ['canonical_id','parent_component','probe_split','cohort','anchor_source_row']})
    return {'root':root,'ready':ready,'meta':pd.DataFrame(meta),'obs':np.concatenate(obs),'now':np.concatenate(now),'endpoint':np.concatenate(endpoints),'endpoint_valid':np.concatenate(endpoint_valid),'state':np.concatenate(state)}
def motion_dataset(features,frame,arm,horizon=50):
    ready,split=truth_contract()
    import pyarrow.parquet as pq
    cols=['label_id','canonical_id','parent_component','probe_split','cohort','anchor_source_row','horizon_steps','arm','dx_m','dy_m','dz_m','horizontal_displacement_m','horizontal_direction','amplitude','text_en','eligible_for_e84']
    df=pq.read_table(R/'fk/fk_motion_labels.parquet',columns=cols,filters=[('arm','=',arm),('horizon_steps','=',horizon),('eligible_for_e84','=',True)]).to_pandas()
    df=df.merge(features['meta'].reset_index(names='feature_index'),on=['canonical_id','parent_component','probe_split','cohort','anchor_source_row'],validate='one_to_one')
    ds=ROBOT_DIRS
    if frame=='robot':
        df=df[(df.horizontal_displacement_m>=.005)&df.horizontal_direction.isin(ROBOT_DIRS)&df.amplitude.isin(AMPLITUDES)].copy()
        df['direction_id']=df.horizontal_direction.map(dict(zip(ROBOT_DIRS,range(8))))
        df['amplitude_id']=df.amplitude.map(dict(zip(AMPLITUDES,range(3))))
        # F amplitudes use 3D displacement; horizontal direction is own-arm base XY.
        df['truth_scope']='own-base FK net horizontal direction + 3D amplitude; not optical pixel truth'
    else:
        # This is deliberately marked coarse, never promoted to precision pixel truth.
        proj=pq.read_table(R/'gt/projection_motion.parquet').to_pandas()
        proj=proj[(proj.arm==arm)&(proj.horizon_steps==horizon)&proj.eligible&proj.direction.isin(SCREEN_DIRS)].copy()
        df=df.merge(proj[['label_id','du_px','dv_px','magnitude_px','direction']],on='label_id',validate='one_to_one')
        df['direction_id']=df.direction.map(dict(zip(SCREEN_DIRS,range(8))))
        # Screen amplitude cutoffs fixed on probe-train only; no use of probe-val distribution.
        train_mag=df.loc[df.probe_split=='probe-train','magnitude_px'].to_numpy();cuts=np.quantile(train_mag,[1/3,2/3])
        df['amplitude_id']=np.searchsorted(cuts,df.magnitude_px.to_numpy(),side='right')
        df['truth_scope']='coarse demo4 legacy projection diagnostic; precision pixel GT not approved'
        df.attrs['pixel_amplitude_train_quantiles']=cuts.tolist()
    ei=0 if horizon==10 else 1;idx=df.feature_index.to_numpy();valid=features['endpoint_valid'][idx,ei].all(-1)
    df=df[valid].copy();idx=df.feature_index.to_numpy()
    start=features['now'][idx];end=features['endpoint'][idx,ei]
    x=np.concatenate([end.reshape(len(df),-1),start.reshape(len(df),-1),(end-start).reshape(len(df),-1)],axis=-1)
    df['label']=(df.direction_id*3+df.amplitude_id).astype(int)
    return df,x,start,end
