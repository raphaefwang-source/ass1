"""Measure a genuinely fresh Python process and an empty compiled-kernel cache."""
import json
import subprocess
import sys
import time
import bench_common as C

def main():
    destination=C.RESULTS/'cold_process_start.json'
    build_dir=C.HERE/'build'/'cold_process_start'
    if destination.exists() or build_dir.exists():
        raise FileExistsError('Preserve this already measured cold-start cache')
    program=('import json; from compiled_edge import compile_backend; '
             'print(json.dumps(compile_backend(build_dir='+repr(str(build_dir))+')))')
    before=C.load_snapshot('cold_process_start')
    started=time.perf_counter()
    process=subprocess.run([sys.executable,'-B','-c',program],cwd=C.HERE,
                           capture_output=True,text=True,check=True)
    wall=time.perf_counter()-started
    info=json.loads(process.stdout)
    if not info['compiled_now']:raise RuntimeError('Cold probe unexpectedly reused a binary')
    C.save(destination,dict(fresh_process_wall_s=wall,backend=info,
        scope='fresh Python process including Python/NumPy/SciPy imports, compiler lookup, true cold compilation and library load; excludes physical table/config setup',
        load_before=before,load_after=C.load_snapshot('cold_process_start_finished')))
    print(dict(fresh_cold_process_s=wall,compiler_s=info['t_compile_s'],load_s=info['t_load_s']),flush=True)

if __name__=='__main__':main()
