#!/usr/bin/env python3
"""Run the actual experiment; optionally join an already running reference job."""
import argparse
import subprocess
import sys
import time
import experiment as X

def call(script,*args):
    X.log('RUN '+script+' '+' '.join(args))
    subprocess.run([sys.executable,str(X.HERE/script),*args],cwd=X.C.REPO,check=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wait-for-references',action='store_true')
    args=ap.parse_args()
    if args.wait_for_references:
        X.log('Waiting for the separately running nine-reference build')
        while len(X.read(X.RESULTS/'references.csv'))!=9:time.sleep(5)
    else:call('build_references.py','--workers','3')
    for cfg in X.configs():X.reference(cfg['name'])
    call('theory.py')
    call('run_optimization.py','all')
    call('analyze_optimization.py','all')
    call('summarize_results.py')
    # Sensitivity can reveal a different M/p tradeoff from the initial grid.
    # Preserve that first pass, refine jointly, and confirm all controls together.
    call('post_sensitivity_refinement.py')
    call('analyze_optimization.py','diagnostics','timing','--refresh-diagnostics')
    call('summarize_results.py')
    X.log('ALL EXPERIMENT STAGES COMPLETE')

if __name__=='__main__':main()
