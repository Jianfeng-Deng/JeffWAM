import time,os,pathlib,json
from common import *
LIMIT={'frozen':6*3600,'lora':4*3600}
class GPUBudget:
    def __init__(self,phase,enabled):
        self.phase=phase;self.enabled=enabled;self.start=time.time();self.path=None
        if enabled:
            self.root=P/'gpu_ledger';self.root.mkdir(exist_ok=True);self.path=self.root/f'{phase}_{os.getpid()}_{int(self.start)}.json'
            self.check(reserve=30);self.record('running')
    def totals(self):
        totals={'frozen':0.,'lora':0.}
        for p in (P/'gpu_ledger').glob('*.json'):
            if p==self.path:continue
            d=json.loads(p.read_text());elapsed=d['elapsed_s'] if d['status']=='finished' else max(d['elapsed_s'],time.time()-d['start_utc'])
            totals[d['phase']]+=elapsed
        if self.enabled:totals[self.phase]+=time.time()-self.start
        return totals
    def check(self,reserve=0):
        if not self.enabled:return
        t=self.totals();assert t[self.phase]+reserve<LIMIT[self.phase],f'{self.phase} GPU budget reached: {t}'
        assert sum(t.values())+reserve<10*3600,f'total GPU budget reached: {t}'
    def record(self,status):
        if not self.enabled:return
        atomic(self.path,{'phase':self.phase,'pid':os.getpid(),'start_utc':self.start,'elapsed_s':time.time()-self.start,'timestamp_brt':brt(),'status':status,'device':os.environ.get('CUDA_VISIBLE_DEVICES')})
        t=self.totals();update(gpu_hours=sum(t.values())/3600,gpu_hours_by_phase={k:v/3600 for k,v in t.items()})
    def heartbeat(self):self.check(reserve=20);self.record('running')
    def close(self):self.record('finished')
