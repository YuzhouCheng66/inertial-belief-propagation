"""Portable benchmark entry point; never writes into the archived results."""
import argparse
import json
import os
from pathlib import Path

NAMES=('smallGrid3D','FR079','Sphere','Cubicle')


def main():
    parser=argparse.ArgumentParser(description='Original local 8+1 Inertial Belief Propagation')
    parser.add_argument('--threads',type=int,default=8,help='precision preparation threads')
    sub=parser.add_subparsers(dest='command',required=True)
    frozen=sub.add_parser('frozen',help='replay a prepared k0 model')
    frozen.add_argument('--dataset',choices=NAMES,required=True)
    frozen.add_argument('--method',choices=('ibp','gbp'),default='ibp')
    frozen.add_argument('--sweeps',type=int,default=8192)
    frozen.add_argument('--tol',type=float,default=1e-6)
    frozen.add_argument('--cache-dir',type=Path)
    frozen.add_argument('--output',type=Path,required=True)
    nonlinear=sub.add_parser('nonlinear',help='20 complete cycles per new linearization')
    nonlinear.add_argument('--dataset',choices=NAMES,required=True)
    nonlinear.add_argument('--method',choices=('ibp','gbp'),default='ibp')
    nonlinear.add_argument('--outer',type=int,default=20)
    nonlinear.add_argument('--data-dir',type=Path)
    nonlinear.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
        os.environ.setdefault(key,'1')
    os.environ.setdefault('NUMBA_NUM_THREADS',str(args.threads))
    import numpy as np
    from numba import set_num_threads, config
    set_num_threads(min(args.threads,config.NUMBA_NUM_THREADS))
    if args.command=='frozen':
        from .benchmarks import load_frozen
        from .solver import solve
        if args.output.exists() and any(args.output.iterdir()):parser.error('--output must be new or empty')
        L,P,pc,offset=load_frozen(args.dataset,args.cache_dir)
        out=solve(L,P,pc,offset,method=args.method,sweeps=args.sweeps,tol=args.tol)
        args.output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output/'trajectory.npz',**{k:out[k] for k in ('eta','x','trace','beta_history')})
        result=out['metrics'];result.update(dataset=args.dataset,scope='frozen k0, precision/basis preparation excluded')
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    else:
        from .nonlinear import run
        result=run(args.dataset,args.method,args.output,outer=args.outer,data_dir=args.data_dir)
    print(json.dumps(result if args.command=='frozen' else {k:result[k] for k in ('name','method','status','accepted_steps','final_cost','totals')},indent=2))
    return 1 if result['status'] in ('diverged','inner_diverged','not_descent','line_search_failed') else 0
