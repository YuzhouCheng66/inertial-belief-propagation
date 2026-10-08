"""Verify six native smoke runs and deterministic fixed-cycle thread behavior."""
from pathlib import Path
import json
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
BIN=ROOT/'build/ibp_parallel'
checks=[]
with tempfile.TemporaryDirectory() as temporary:
    tmp=Path(temporary)
    for case in ('check_0','check_1','check_2'):
        for method in (0,1):
            reference=None
            for threads in (1,4):
                prefix=tmp/f'{case}_m{method}_t{threads}'
                process=subprocess.run([str(BIN),str(ROOT/f'data/{case}.bin'),str(method),str(threads),
                    '40','40',str(prefix),'-'],check=True,text=True,capture_output=True)
                result=json.loads(process.stdout.strip().splitlines()[-1])
                state=prefix.with_suffix('.state').read_bytes()
                history=prefix.with_suffix('.history').read_bytes()
                if reference is None:reference=(state,history)
                else:assert (state,history)==reference,(case,method,'thread-dependent trajectory')
                checks.append(dict(case=case,method=method,threads=threads,sweeps=result['sweeps'],
                    evals=result['evals'],corrections=result['corrections']))
print(json.dumps(dict(fixed_cycle_thread_determinism_pass=True,runs=checks),indent=2))
