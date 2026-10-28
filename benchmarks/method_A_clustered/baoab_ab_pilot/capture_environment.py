"""Save local execution environment before experiments in a source-only checkout."""
from evidence import HERE,RESULTS,BASE,save
import datetime
import importlib
import os
import platform
import shutil
import subprocess
import sys

def main():
    target=RESULTS/'environment_initial.json'
    if target.exists():raise FileExistsError('Preserve existing initial environment evidence')
    root=HERE.parents[2]
    def run(args):return subprocess.check_output(args,cwd=root,text=True).strip()
    data=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cwd=str(root),branch=run(['git','branch','--show-current']),
        platform=platform.platform(),machine=platform.machine(),python=sys.version,
        python_executable=sys.executable,cpu_count=os.cpu_count(),execution='local Mac only',stages=[0,1])
    if platform.system()!='Darwin':raise RuntimeError('This experiment is registered for the local Mac')
    for name in ('numpy','scipy','matplotlib','threadpoolctl','psutil','numba'):
        try:data[name]=getattr(importlib.import_module(name),'__version__','available')
        except Exception as e:data[name]={'error':repr(e)}
    data['macos']=run(['sw_vers']);data['clang']=run(['clang','--version'])
    data['physical_memory_bytes']=run(['sysctl','-n','hw.memsize'])
    data['cpu']=run(['sysctl','-n','machdep.cpu.brand_string'])
    save(target,data)
    print('Saved',target)

if __name__=='__main__':main()
