import sys,json,subprocess,concurrent.futures
sys.dont_write_bytecode=True
from e103d_runtime import *
from e103d_cache03 import packets
plan=json.loads((CACHE/'plan.json').read_text());result=[]
for r in plan['records']:
 p=CACHE/'features'/r['canonical_id']/'receipt.json'
 if not p.exists() or r['source']!='abc':continue
 d=json.loads(p.read_text())
 if d.get('decoder_stats'):continue
 raws=packets(r)
 def one(k):
  out=subprocess.run(['ffprobe','-v','error','-threads','1','-f','h264','-show_frames','-show_entries','frame=pict_type','-of','json','-i','pipe:0'],input=raws[k],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True);frames=json.loads(out.stdout)['frames'];assert len(frames)==len(r['timestamps'][k]);b=sum(f.get('pict_type')=='B' for f in frames);assert b==0;return dict(camera=k,frames=len(frames),b_frames=b)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:stats=list(pool.map(one,range(3)))
 result.append(dict(cid=r['canonical_id'],stats=stats));print('audited',len(result),flush=True)
atomic(CONTROL/'EARLY_DECODE_AUDIT.json',dict(status='passed',episodes=len(result),records=result,brt=brt()));report('## cache02 早期帧序补审计\n'+json.dumps(dict(episodes=len(result),status='passed',all_B_frames_zero=True,packet_decode_count_equal=True,CPU_only=True),ensure_ascii=False))
