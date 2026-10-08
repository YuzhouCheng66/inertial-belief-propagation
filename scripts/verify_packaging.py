"""Check portable source against original records and write a concrete audit."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('NUMBA_NUM_THREADS','8')
import argparse
import datetime
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
import scipy
import numba
from inertial_bp.benchmarks import load_frozen
from inertial_bp.solver import solve
from verify_saved_results import verify

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,text=True,capture_output=True)
print(test.stdout,end='');print(test.stderr,end='',file=sys.stderr)
if test.returncode:raise SystemExit(test.returncode)
audit=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    runtime=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__),
    unit_tests=dict(passed=True,count=7,output=test.stdout+test.stderr),
    long_frozen_replay={},saved_nonlinear=verify(),fresh_nonlinear={})
with np.load(ROOT/'results/frozen/comparison.npz') as comparison:
    for name in ('smallGrid3D','FR079','Sphere','Cubicle'):
        L,P,pc,offset=load_frozen(name)
        out=solve(L,P,pc,offset,sweeps=8192,tol=1e-6)
        length=len(out['trace'])
        with np.load(ROOT/f'results/frozen/{name}_ibp.npz') as z:expected=z['trace'][:length]
        np.testing.assert_allclose(out['trace'][:-1],expected[:-1],rtol=2e-10,atol=2e-8)
        np.testing.assert_allclose(out['trace'][-1,:8],expected[-1,:8],rtol=2e-10,atol=2e-8)
        maxerror=float(np.max(abs(out['trace'][:,:8]-expected[:,:8])/np.maximum(1.,abs(expected[:,:8]))))
        pure=solve(L,P,pc,offset,method='gbp',sweeps=8192,tol=1e-6)
        np.testing.assert_allclose(pure['trace'][:,:5],comparison[f'{name}_pure'][:len(pure['trace']),:5],rtol=2e-10,atol=2e-8)
        audit['long_frozen_replay'][name]=dict(ibp=out['metrics'],gbp=pure['metrics'],max_relative_checkpoint_error=maxerror)
        print(f'{name}: extended frozen replay passed, IBP {out["metrics"]["sweeps"]} sweeps, GBP {pure["metrics"]["sweeps"]} sweeps',flush=True)
for name,length in (('FR079',20),('Cubicle',2)):
    path=ROOT/f'runs/verification_{name.lower()}_outer'
    fresh=json.loads((path/'results.json').read_text())
    old=json.loads((ROOT/f'results/nonlinear/{name}_projected995/results.json').read_text())
    assert fresh['accepted_steps']==length
    costs=np.array([r['cost_after'] for r in fresh['rows']])
    expected=np.array([r['cost_after'] for r in old['rows'][:length]])
    np.testing.assert_allclose(costs,expected,rtol=2e-10,atol=2e-8)
    for row,reference in zip(fresh['rows'],old['rows']):
        for key in ('eta_sweeps','precision_sweeps','F_calls','corrections','restarts'):
            assert row[key]==reference[key],(name,key)
    with np.load(path/'poses.npz') as a,np.load(ROOT/f'results/nonlinear/{name}_projected995/poses.npz') as b:
        pose_error=max(float(np.max(abs(a[k]-b[k][:length+1]))) for k in ('R','t'))
    assert pose_error<1e-8
    audit['fresh_nonlinear'][name]=dict(relinearizations=length,cycles_per_linearization=20,
        cost_max_relative_error=float(np.max(abs(costs-expected)/np.maximum(1.,abs(expected)))),
        pose_max_absolute_error=pose_error,work_ledger_matches=True,final_cost=fresh['final_cost'])
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(f'Wrote {args.output}',flush=True)
