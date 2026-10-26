import sys,time,threading,tempfile,types
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
import final_davis as m
now=[];lock=threading.Lock()
def emit(tag):
    with lock:now.append((tag,time.perf_counter()))
old_plan,old_cpu,old_lane,old_run,old_fargs=m._plan,m._matlab_task,m._sequential_lane,m.run,m.frozen_args
try:
    def plan(source,videos):return ([('camel',.2,'KBR'),('camel',.2,'TNN'),('camel',.2,'SALTS'),('camel',.3,'KBR')], [('camel__sr020','DRO-TFF')])
    def cpu(*args):
        v,sr,method=args[1:4]
        emit('cpu-start-'+method+str(sr));time.sleep(.23);emit('cpu-end-'+method+str(sr))
        return v,method,0,'test'
    def lane(*args):
        emit('gpu-start');time.sleep(.28);emit('gpu-end')
    def fake_run(cmd):emit('evaluate')
    def fargs(*args):return ['dryrun_report']
    m._plan=plan;m._matlab_task=cpu;m._sequential_lane=lane;m.run=fake_run;m.frozen_args=fargs
    with tempfile.TemporaryDirectory() as d:
        args=types.SimpleNamespace(baseline_mode='missing',jobs=4,matlab_threads=4,dry_run=False)
        m.run_heterogeneous(Path(d),Path(d),Path(d),Path(d),['camel'],args)
    get=lambda prefix: [t for n,t in now if n.startswith(prefix)]
    starts=get('cpu-start');ends=get('cpu-end');gpu=get('gpu-');evaluation=get('evaluate')
    assert len(starts)==len(ends)==4 and len(gpu)==2 and len(evaluation)==1
    assert max(starts)<min(ends),'CPU tasks did not overlap'
    assert min(starts)<gpu[0]<max(ends),'GPU lane did not overlap CPU jobs'
    assert evaluation[0]>max([*ends,*gpu]),'Evaluation began before all tasks finished'
    print('V15 SCHEDULER SMOKE PASS: 4 parallel CPU jobs + 1 sequential GPU lane overlap; report waits for all')
finally:
    m._plan,m._matlab_task,m._sequential_lane,m.run,m.frozen_args=old_plan,old_cpu,old_lane,old_run,old_fargs
