"""Per-epsilon same-batch frontier; preserve both original timing passes."""
import study as X

def main():
    primary=X.read(X.RESULTS/'primary_timing'/'accuracy_time_frontier.csv')
    confirm=X.read(X.RESULTS/'accuracy_time_frontier.csv')
    if not primary or not confirm:raise RuntimeError('Need both completed timing passes')
    rows=[]
    for eps in (1e-3,1e-4,1e-5,1e-6):
        batch='main_confirmation' if eps==1e-4 else 'primary'
        source=confirm if eps==1e-4 else primary
        rows += [dict(r,timing_batch=batch,timing_source=('accuracy_time_frontier.csv' if eps==1e-4 else 'primary_timing/accuracy_time_frontier.csv'))
                 for r in source if r['epsilon']==eps]
    X.write(X.RESULTS/'frontier.csv',rows)

if __name__=='__main__':main()
