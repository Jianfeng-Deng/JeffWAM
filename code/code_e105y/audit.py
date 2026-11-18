import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,hashlib,collections,datetime,math,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');C=T/'code_e105y';R=Path((C/'run_path.txt').read_text().strip());P=T/'notes/e105y_yam_feasibility_20261115.md'
def dump(name,d):
 p=R/name;assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def append(s):
 with P.open('a') as f:f.write('\n'+s+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((T/'data/e103d_cache/plan.json').read_text());ys=[r for r in plan['records'] if r['source']=='yam'];abc=[r for r in plan['records'] if r['source']=='abc'];H=[10,25,50,100]
# Same timestamp gates as E103-D, independently selectable per horizon, no import with side effects.
def gate(ts,a,hs):
 start=max(t[0] for t in ts);off=np.array([-24,-18,-12,-6,0]+hs);times=start+np.rint((a+off)*1e9/30).astype('int64');selected=[]
 for t in ts:
  dt=np.diff(t);fps=(len(t)-1)*1e9/(t[-1]-t[0])
  if not 20<=fps<=65 or dt.max()>1e9 or np.mean(dt>1e8)>.005:return False
  prev=np.searchsorted(t,times,side='right')-1
  if np.any(prev[:5]<0) or times[-1]>t[-1]:return False
  ix=np.clip(prev,0,len(t)-1);nxt=np.clip(prev+1,0,len(t)-1);near=np.where(abs(t[ix]-times)<=abs(t[nxt]-times),ix,nxt);j=np.r_[ix[:5],near[5:]]
  age=times[:5]-t[j[:5]];err=abs(times[5:]-t[j[5:]]);tol=min(50e6,max(.1e6,np.median(dt)/2+.01e6))
  if age.min()<0 or age.max()>1e8 or err.max()>tol:return False
  if any(not(times[-1]<t[k] or times[0]>t[k+1]) for k in np.where(dt>1e8)[0]):return False
  selected.append(t[j])
 return np.ptp(np.stack(selected),axis=0).max()<=5e7
rows=[];meta=[];states=collections.Counter();video_stats=[];evidence=[];all_windows=[]
for r in ys:
 o=r['original'];raw=Path(o['representative_path']);m=json.loads((raw/'metadata.json').read_text());meta.append(m);ts=[np.rint(np.load(raw/(k+'-timestamp.npy'),allow_pickle=False)*1e6).astype('int64') for k in ['top','left','right']]
 assert all(np.array_equal(a,np.array(b,dtype='int64')) for a,b in zip(ts,r['timestamps']))
 n=int((min(t[-1] for t in ts)-max(t[0] for t in ts))*30//10**9)+1
 sparse={h:sum(gate(ts,int(a),[h]) for a in np.arange(24,n-h,30)) for h in H}
 all4=sum(gate(ts,int(a),H) for a in np.arange(24,n-100,30));assert all4==len(r['windows'])
 candidates=np.unique(np.linspace(24,n-53,16).astype(int)//4*4);fixed=[]
 for a in candidates:
  if gate(ts,int(a),[10,25,50]):fixed.append(int(a))
  if len(fixed)==8:break
 fixedh={h:sum(gate(ts,a,[h]) for a in fixed) for h in H}
 for a in fixed:all_windows.append(dict(cid=r['canonical_id'],anchor=a,valid={str(h):bool(gate(ts,a,[h])) for h in H}))
 for f in raw.glob('*.npy'):
  z=np.load(f,allow_pickle=False);states[(f.name,str(z.shape[1:]),str(z.dtype),bool(np.isfinite(z).all()))]+=1
 for k in ['top','left','right']:
  f=raw/(k+'-images-rgb.mp4');v=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration','-of','json',str(f)]))['streams'][0];video_stats.append(dict(cid=r['canonical_id'],camera=k,**v))
 day=r['canonical_id'][4:12];clock=r['canonical_id'][13:19];dt=datetime.datetime.strptime(day+clock,'%Y%m%d%H%M%S')
 rows.append(dict(cid=r['canonical_id'],date=day,start=dt.isoformat(),frames=o['frames'],nominal_seconds=o['frames']/30,common_span_s=(min(t[-1] for t in ts)-max(t[0] for t in ts))/1e9,actual_fps=[(len(t)-1)*1e9/(t[-1]-t[0]) for t in ts],sparse=sparse,all4=all4,fixed=fixedh,anchors=fixed,raw=str(raw),meta_sha=sha(raw/'metadata.json')))
# Date sessions are conservative working units. >5-minute inactive interval defines optimistic blocks, never single clips.
rows.sort(key=lambda r:r['start']);relax=0;prev=None
for r in rows:
 dt=datetime.datetime.fromisoformat(r['start'])
 if prev is None or r['date']!=prev['date'] or (dt-datetime.datetime.fromisoformat(prev['start'])).total_seconds()-prev['common_span_s']>300:relax+=1
 r['conservative']='day_'+r['date'];r['relaxed']='block_'+str(relax);prev=r
units={}
for mode in ['conservative','relaxed']:
 groups=[]
 for name in sorted({r[mode] for r in rows}):
  rr=[r for r in rows if r[mode]==name];ww=[w for w in all_windows if w['cid'] in {r['cid'] for r in rr}];ids=np.linspace(0,len(ww)-1,min(8,len(ww))).astype(int);chosen=[ww[i] for i in ids]
  groups.append(dict(parent=name,clips=len(rr),seconds=sum(r['nominal_seconds'] for r in rr),clip_fixed={h:sum(r['fixed'][h] for r in rr) for h in H},sparse={h:sum(r['sparse'][h] for r in rr) for h in H},dev_fixed={h:sum(w['valid'][str(h)] for w in chosen) for h in H},selected=chosen,cids=[r['cid'] for r in rr]))
 units[mode]=groups
urdf=Path('/mnt/workspace/colleague-f/yam_model/yam.urdf');root=ET.fromstring(urdf.read_text());joints=[dict(name=j.attrib['name'],type=j.attrib['type'],origin=(j.find('origin').attrib if j.find('origin') is not None else {}),limit=(j.find('limit').attrib if j.find('limit') is not None else {})) for j in root.findall('joint')]
dump('yam_inventory.json',rows);dump('camera_probe.json',video_stats);dump('state_urdf.json',dict(arrays=[dict(name=k[0],tail_shape=k[1],dtype=k[2],finite=k[3],clips=v) for k,v in states.items()],urdf=str(urdf),urdf_sha=sha(urdf),joints=joints,metadata=meta));dump('parent_groups.json',units);dump('fixed_windows.json',all_windows)
dur=np.array([r['nominal_seconds'] for r in rows]);total={h:sum(r['fixed'][h] for r in rows) for h in H};sparse={h:sum(r['sparse'][h] for r in rows) for h in H};stats=dict(clips=len(rows),frames=sum(r['frames'] for r in rows),nominal_seconds=float(dur.sum()),duration_quantiles_s=np.quantile(dur,[0,.25,.5,.75,1]).tolist(),duration_mean_s=float(dur.mean()),common_span_s=sum(r['common_span_s'] for r in rows),actual_fps_range=[min(v for r in rows for v in r['actual_fps']),max(v for r in rows for v in r['actual_fps'])],clip_fixed=total,sparse=sparse,all4=sum(r['all4'] for r in rows))
dump('summary.json',stats)
append('## 实测清单（完成）\n'+json.dumps(stats,ensure_ascii=False)+'\n\n|片段|帧数|标称秒|相机共同时间跨度秒|固定窗 +10/+25/+50/+100|保守/宽松组|\n|---|---:|---:|---:|---|---|\n'+'\n'.join('|'+r['cid']+'|'+str(r['frames'])+'|'+f"{r['nominal_seconds']:.3f}"+'|'+f"{r['common_span_s']:.3f}"+'|'+'/'.join(str(r['fixed'][h]) for h in H)+'|'+r['conservative']+'/'+r['relaxed']+'|' for r in rows))
append('## parent 归并及固定窗口实测（完成）\n43 个 canonical 去重录制 ID ≠43 个独立 parent。29段于2026-09-22、14段于2026-10-16；同任务、同类桌面、操作者身份字段0/43。保守工作口径按录制日合并为2个；宽松口径按相邻片段结束至下一片开始空档>300秒另开块，共'+str(len(units['relaxed']))+'个（300秒是本轮预先用于敏感性分析的元数据规则，不是独立性证明，不来自模型评分）。日期/空档仍不能证明换场景/操作者，严格同场景敏感性下全部合并为1个，独立 dev 不成立。不可把逐episode当独立 parent。\n\nE102-X/E93口径：16个固定分位候选、向下取4帧、前8合法窗；历史-24/-18/-12/-6/0；先要求+10/+25/+50合法，再逐窗mask +100。本轮用实际相机共同30Hz时间网格与E103-D age/skew/gap闸门；只有host receipt近似，无独立曝光/接收双时钟，不冒称完全等价Piper时钟合同。先生成各片段固定窗，再在每个归并parent池按时间均匀取最多8个作为推荐dev；详细锚点已锁 fixed_windows/parent_groups JSON。跨片段不拼接。\n\n|口径|parent|片段|clip固定窗四时域|parent限制后dev固定窗四时域|稀疏1秒窗四时域|\n|---|---|---:|---|---|---|\n'+'\n'.join('|'+mode+'|'+g['parent']+'|'+str(g['clips'])+'|'+'/'.join(str(g['clip_fixed'][h]) for h in H)+'|'+'/'.join(str(g['dev_fixed'][h]) for h in H)+'|'+'/'.join(str(g['sparse'][h]) for h in H)+'|' for mode,gg in units.items() for g in gg))
print(json.dumps(dict(stats=stats,groups={m:[{k:v for k,v in g.items() if k not in ['selected','cids']} for g in gs] for m,gs in units.items()}),ensure_ascii=False))
(R/'STATUS.json').write_text(json.dumps(dict(status='inventory_complete',gpu_seconds=0)))
