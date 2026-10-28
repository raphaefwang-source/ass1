"""Source-only reproduction entry: preserve pre-experiment Mac environment."""
from joint_common import *
import datetime
import importlib
import platform
import shutil
import subprocess


def main():
    target=RESULTS/'environment_initial.json'
    if target.exists():raise FileExistsError('Preserve the original pre-experiment environment')
    if platform.system()!='Darwin':raise RuntimeError('This experiment is registered for local Mac execution')
    data=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cwd=str(REPO),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        head_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        platform=platform.platform(),machine=platform.machine(),python=sys.version,python_executable=sys.executable,
        execution='local Mac only',scope='two-tier frozen-accuracy-constrained dynamic joint pilot',
        free_disk_bytes=shutil.disk_usage(REPO).free)
    for name in ('numpy','scipy','matplotlib','threadpoolctl','psutil'):
        data[name]=importlib.import_module(name).__version__
    for args,key in [(['sysctl','-n','hw.memsize'],'physical_memory_bytes'),(['clang','--version'],'clang'),(['sw_vers'],'macos')]:
        data[key]=subprocess.check_output(args,text=True).strip()
    snapshot=HERE/'baseline_snapshot.json'
    if not snapshot.exists():
        tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=REPO,text=True).splitlines()
        save(snapshot,dict(base_sha=BASE,tracked_file_sha256={name:sha(REPO/name) for name in tracked}))
    save(target,data)
    register_domain()
    print('Saved local initial environment and registered readonly starts:',target)


if __name__=='__main__':main()
