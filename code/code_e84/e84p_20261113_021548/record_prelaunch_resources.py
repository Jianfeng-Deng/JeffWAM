from common import *
from contract import sha
ledger=list((P/'gpu_ledger').glob('*.json'))
start=json.loads(ledger[0].read_text())['start_utc']
admission=json.loads((P/'gpu_shared_admission.json').read_text())
atomic(P/'gpu_all_cards_prelaunch_record.json',{'record_written_brt':brt(),'observation_interval_brt':['2026-11-13T02:31:21.556160-03:00',datetime.datetime.fromtimestamp(start,datetime.timezone(datetime.timedelta(hours=-3))).isoformat()],'source':'Exact nvidia-smi stdout retained in execution tool transcript, after start_resume.py prepared and before detached extract started. This file transcribes existing evidence; it is not a new historical timestamp claim.','command':'nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader','devices':[{'index':i,'free_mib':free,'utilization_percent':u} for i,free,u in [(0,8194,0),(1,75368,99),(2,75370,80),(3,75409,100),(4,89528,0),(5,75239,99),(6,12801,100),(7,31666,100)]],'selected':4,'estimated_peak_mib':8192,'factor':1.5,'required_free_mib':12288,'selected_admission_record':str(P/'gpu_shared_admission.json'),'selected_admission_sha256':sha(P/'gpu_shared_admission.json')})
print('PRELAUNCH_ALL_CARD_EVIDENCE_PRESERVED')
