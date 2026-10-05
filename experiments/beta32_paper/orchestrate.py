#!/usr/bin/env python3
"""Continue all original paper experiments and write/commit completed results."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from run_study import write_json

PRIMARY='bal9419_d8_s0_b32'
LOCK=threading.Lock()

def command(script,*args):
    env=os.environ.copy()
    env.setdefault('MPLCONFIGDIR',str(HERE/'cache/matplotlib'))
    env.setdefault('LOKY_MAX_CPU_COUNT','8')
    env.setdefault('OMP_NUM_THREADS','2')
    env.setdefault('OPENBLAS_NUM_THREADS','2')
    env.setdefault('VECLIB_MAXIMUM_THREADS','2')
    logdir=HERE/'cache/logs';logdir.mkdir(parents=True,exist_ok=True)
    name=script.replace('.py','')+'_'+ '_'.join(str(a).replace('/','-') for a in args)+'.log'
    with open(logdir/name,'a') as log:
        subprocess.run([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)

def refresh():
    with LOCK:command('report.py')

def local_commit(message):
    # Human explicitly requested local commits. Scope is this new folder only.
    with LOCK:
        subprocess.run(['git','add','--','experiments/beta32_paper'],cwd=ROOT,check=True)
        changed=subprocess.run(['git','diff','--cached','--quiet','--','experiments/beta32_paper'],cwd=ROOT)
        if changed.returncode==1:
            subprocess.run(['git','commit','--only','-m',message,'--','experiments/beta32_paper'],cwd=ROOT,check=True)
        elif changed.returncode!=0:raise RuntimeError('Could not inspect staged experiment files')

def analysis(stem,primary=False):
    command('evaluate_study.py','primary' if primary else 'scan','--ids',stem)
    refresh()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',choices=['mps','cuda','cpu'])
    ap.add_argument('--externally-running',nargs='*',default=[])
    ap.add_argument('--no-commit',action='store_true')
    args=ap.parse_args()
    lock=open(HERE/'driver.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A study driver is already running')
    plan=json.loads((HERE/'plan.json').read_text())
    milestones=HERE/'milestones.json'
    state=json.loads(milestones.read_text()) if milestones.exists() else {}
    stop=threading.Event()
    write_json(HERE/'execution.json',{'status':'running','pid':os.getpid(),'started':time.time(),
        'device_requested':args.device,'externally_running':args.externally_running})
    def trainer():
        for task in plan:
            if stop.is_set():return
            stem=task['id'];done=HERE/'runs'/stem/'complete.json'
            if done.exists():continue
            if stem in args.externally_running:
                print('Waiting for existing training',stem,flush=True)
                while not done.exists():
                    if stop.is_set():return
                    time.sleep(5)
            else:
                print('Training',stem,flush=True)
                opts=['train','--ids',stem]
                if args.device:opts+=['--device',args.device]
                command('run_study.py',*opts)
            refresh()
    def evaluator():
        pending={t['id']:t for t in plan}
        while pending:
            if stop.is_set():return
            ready=[t for t in plan if t['id'] in pending and (HERE/'runs'/t['id']/'complete.json').exists()]
            if not ready:time.sleep(5);continue
            for task in ready:
                stem=task['id'];print('Evaluating',stem,flush=True)
                analysis(stem,primary=task['family'] in ['main','paired_control'])
                pending.pop(stem)
                main_complete=all((HERE/'results'/t['id']/'scan_metrics.json').exists()
                    for t in plan if t['family'] in ['main','paired_control'])
                if main_complete and not state.get('main'):
                    command('figures.py');refresh()
                    if not args.no_commit:local_commit('feat(beta32): report main paper comparison')
                    state['main']=True;write_json(milestones,state)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(trainer),pool.submit(evaluator)]
            for future in futures:
                future.add_done_callback(lambda f:stop.set() if f.exception() else None)
            for future in concurrent.futures.as_completed(futures):future.result()
        # Supplementary analyses retain the paper's full budgets. A failure is
        # recorded, not converted into a successful or partially fabricated result.
        for stage in ['contamination','cluster_scan','stability','two_sample']:
            print('Supplementary',stage,flush=True)
            command('evaluate_study.py',stage,'--ids',PRIMARY);refresh()
        command('figures.py');refresh()
        coverage=json.loads((HERE/'coverage.json').read_text())
        if not coverage['full_numerical_reproduction_complete']:
            raise RuntimeError('Coverage audit did not confirm completion')
        if not args.no_commit:local_commit('feat(beta32): complete paper reproduction')
        write_json(HERE/'execution.json',{'status':'complete','pid':os.getpid(),'finished':time.time(),
                                         'coverage':coverage})
        print('FULL NUMERICAL REPRODUCTION COMPLETE',flush=True)
    except BaseException as exc:
        write_json(HERE/'execution.json',{'status':'failed','pid':os.getpid(),'failed':time.time(),
                                         'error':str(exc)})
        refresh()
        raise

if __name__=='__main__':main()
