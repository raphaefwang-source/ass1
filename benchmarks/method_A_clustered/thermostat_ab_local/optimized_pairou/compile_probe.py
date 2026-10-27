"""Cold compilation and fresh-process cache-load costs, separately recorded."""
import json
import subprocess
import sys
import time
import bench_common as C
from compiled_edge import compile_backend

def main():
    path=C.RESULTS/'compilation.json'
    if path.exists():raise FileExistsError(path)
    t=time.perf_counter();cold=compile_backend(force_rebuild=True);cold_total=time.perf_counter()-t
    program='import json; from compiled_edge import compile_backend; print(json.dumps(compile_backend()))'
    t=time.perf_counter()
    warm_process=subprocess.run([sys.executable,'-B','-c',program],cwd=C.HERE,capture_output=True,text=True,check=True)
    process_total=time.perf_counter()-t
    warm=json.loads(warm_process.stdout)
    C.save(path,dict(cold=cold,cold_entry_total_s=cold_total,fresh_process_cache_load=warm,
                    fresh_process_total_s=process_total,numba=C.environment()['numba_status'],
                    interpretation='cold compiler subprocess vs cached library load; fresh-process total also includes Python/NumPy imports'))
    C.save('environment.json',C.environment())
    print(dict(cold_compile_s=cold['t_compile_s'],cold_entry_s=cold_total,cached_load_s=warm['t_load_s'],fresh_process_total_s=process_total),flush=True)

if __name__=='__main__':main()
