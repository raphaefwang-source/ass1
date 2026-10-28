"""Reproducible local environment record, captured before numerical work."""
from matched_common import *
import datetime
import importlib
import platform
import subprocess

def main():
    path=RESULTS/'environment_initial.json'
    if path.exists():raise FileExistsError('Preserve existing initial environment')
    if platform.system()!='Darwin':raise RuntimeError('This pilot is registered for local Mac execution')
    data=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cwd=str(REPO),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        platform=platform.platform(),machine=platform.machine(),python=sys.version,python_executable=sys.executable,
        execution='local Mac only',scope='matched frozen-transition accuracy dynamic-cost pilot',canonical_preparation_reused=True)
    for name in ('numpy','scipy','matplotlib','threadpoolctl','psutil'):
        data[name]=importlib.import_module(name).__version__
    for args,key in [(['sysctl','-n','hw.memsize'],'physical_memory_bytes'),(['clang','--version'],'clang'),(['sw_vers'],'macos')]:
        data[key]=subprocess.check_output(args,text=True).strip()
    save(path,data);print('Saved initial environment',path)

if __name__=='__main__':main()
