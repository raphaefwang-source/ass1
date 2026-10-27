"""Serial orchestration with resumable accuracy/search checkpoints.

Existing historical data are never written. Timing is skipped if already
complete, and a partial timing run must be resumed in a NEW output folder.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results';RESULTS.mkdir(exist_ok=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--from-stage',choices=['tests','sweeps','convergence','xi','coarse','local','boundary','table-audit','merge-diagnostics','regimes','timing','cross-winner','archive-primary','confirmation','frontiers','plots','report','audit-final','report-refresh'],default='tests')
    a=ap.parse_args()
    stages=[('tests',['test_grid_psd.py']),('sweeps',['run_accuracy.py','sweeps','--workers',str(a.workers)]),
        ('convergence',['run_accuracy.py','convergence','--workers',str(a.workers)]),('xi',['run_xi_sweep.py','--workers',str(a.workers)]),
        ('coarse',['run_search.py','coarse']),('local',['run_search.py','local']),('boundary',['run_search.py','boundary']),
        ('table-audit',['table_audit.py']),('merge-diagnostics',['merge_diagnostics.py']),('regimes',['analyze_regimes.py']),('timing',['run_timing.py']),
        ('cross-winner',['cross_winner.py']),('archive-primary',['archive_primary_timing.py']),('confirmation',['run_timing.py','--main-confirmation']),('frontiers',['merge_frontiers.py']),('best-table-audit',['best_table_audit.py']),
        ('plots',['plot_results.py']),('report',['report_generator.py']),
        ('audit-final',['finalize_audit.py']),('report-refresh',['report_generator.py'])]
    active=False
    for stage,args in stages:
        active=active or stage==a.from_stage
        if not active:continue
        if stage=='timing' and (RESULTS/'timing_protocol.json').exists():continue
        if stage=='archive-primary' and (RESULTS/'primary_timing').exists():continue
        if stage=='confirmation' and (RESULTS/'timing_protocol.json').exists():
            if json.loads((RESULTS/'timing_protocol.json').read_text()).get('timing_batch')=='main_confirmation':continue
        command=[sys.executable,'-B',str(HERE/args[0]),*args[1:]]
        with (RESULTS/'execution_commands.jsonl').open('a') as f:
            f.write(json.dumps(dict(started_unix=time.time(),stage=stage,command=command),ensure_ascii=False)+'\n')
        with (RESULTS/f'{stage}_run.log').open('a') as log:
            subprocess.run(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(stage,'complete',flush=True)

if __name__=='__main__':main()
