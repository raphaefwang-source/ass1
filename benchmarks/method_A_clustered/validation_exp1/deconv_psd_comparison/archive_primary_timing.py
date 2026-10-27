"""Preserve the entire first timing pass before independent main confirmation."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;R=HERE/'results'
FILES=('serial_prescreen_raw.csv','timing_plan.json','timed_candidates.csv','table_setup_raw.csv',
 'config_setup_raw.csv','memory.csv','final_timing.csv','final_timing_raw.csv','best_configuration.json',
 'best_configuration.csv','accuracy_time_frontier.csv','frontier.csv','paired_speed_ratios.csv',
 'components_raw.csv','common_kernel_setup_raw.csv','system_load.json','timing_protocol.json',
 'timing_progress.json','timing_run.log')

def main():
    dest=R/'primary_timing'
    if dest.exists():raise RuntimeError('Primary archive exists; do not overwrite')
    if not (R/'timing_protocol.json').exists():raise RuntimeError('First pass is not complete')
    dest.mkdir();manifest={}
    for name in FILES:
        p=R/name
        if p.exists():
            manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            p.rename(dest/name)
    (R/'primary_archive_manifest.json').write_text(json.dumps(dict(sha256=manifest,note='All first-pass artifacts moved without modification; final main comparison uses a fresh common confirmation batch'),indent=2)+'\n')

if __name__=='__main__':main()
