"""CPU supervisor: terminate only verified E86-owned GPU workers before hard walltime."""
import os,time,json,signal,datetime
from pathlib import Path
from .common import *
def main():
    path=Path(os.environ.get('E86_WATCHDOG_STATUS',str(C/'artifacts/watchdog_v1_status.json')));driver_path=Path(os.environ.get('E86_DRIVER_STATUS',str(C/'artifacts/driver_v2_status.json')));ledger=C/'artifacts/gpu_ledger.json'
    while True:
        d=json.loads(ledger.read_text());active=d.get('active');driver=json.loads(driver_path.read_text())
        if active:
            elapsed=time.time()-datetime.datetime.fromisoformat(active['started_brt']).timestamp();total=d['elapsed_s']+elapsed;limit=min(active['limit_s'],10800-d['elapsed_s']);pid=active['pid']
            if elapsed>=limit-20:
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes();cwd=Path(f'/proc/{pid}/cwd').resolve()
                if b'e86.' not in cmd or cwd!=C.resolve() or Path(f'/proc/{pid}').stat().st_uid!=os.getuid():raise RuntimeError('Refuse unverified PID')
                atomic(path,dict(status='stopping_verified_own_worker',pid=pid,reason='phase/global walltime',elapsed_s=elapsed,total_s=total,updated_brt=brt()))
                os.kill(pid,signal.SIGTERM)
                for _ in range(5):
                    if not Path(f'/proc/{pid}').exists():break
                    time.sleep(1)
                if Path(f'/proc/{pid}').exists():os.kill(pid,signal.SIGKILL);time.sleep(1)
                # Worker may have closed normally; only reconcile an unchanged abandoned active segment.
                latest=json.loads(ledger.read_text())
                if latest.get('active',{}).get('pid')==pid:
                    end=time.time()-datetime.datetime.fromisoformat(active['started_brt']).timestamp();seg=latest.pop('active');seg.update(elapsed_s=end,finished_brt=brt(),stop_reason='verified E86 watchdog hard walltime');latest['segments'].append(seg);latest['elapsed_s']+=end;atomic(ledger,latest)
                atomic(path,dict(status='stopped_own_worker',pid=pid,updated_brt=brt()));return
        elif driver.get('status') in ('completed','stopped'):
            atomic(path,dict(status='finished_no_worker_killed',driver_status=driver['status'],gpu_hours=d['elapsed_s']/3600,updated_brt=brt()));return
        atomic(path,dict(status='monitoring',active_phase=active.get('phase') if active else None,gpu_hours_including_active=(d['elapsed_s']+elapsed)/3600 if active else d['elapsed_s']/3600,updated_brt=brt()));time.sleep(2)
if __name__=='__main__':main()
