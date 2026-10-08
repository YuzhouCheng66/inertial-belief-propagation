"""Recompute all archived nonlinear costs and Gaussian warp identities."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('NUMBA_NUM_THREADS','8')
import json
from pathlib import Path
import numpy as np
from inertial_bp.nonlinear import load_case
from inertial_bp.warp import warp
from inertial_bp.models import log_pose,jr_inv
from inertial_bp.lie import retract

ROOT=Path(__file__).resolve().parents[1]


def verify():
    checks={}
    for name in ('FR079','Cubicle'):
        g=load_case(name)
        for method in ('projected995','GBP'):
            root=ROOT/f'results/nonlinear/{name}_{method}'
            data=json.loads((root/'results.json').read_text())
            with np.load(root/'poses.npz') as z:R,t=z['R'],z['t']
            accepted=[r for r in data['rows'] if 'cost_after' in r]
            costs=np.array([g.evaluate(rr,tt) for rr,tt in zip(R,t)])
            expected=np.array([data['initial_cost']]+[r['cost_after'] for r in accepted])
            assert len(R)==data['accepted_steps']+1==len(expected)
            error=float(np.max(abs(costs-expected)/np.maximum(1.,abs(expected))))
            assert error<1e-12
            assert np.all(np.diff(costs)<=1e-12*np.maximum(1.,abs(costs[:-1])))
            assert np.max(abs(R[:,0]-np.eye(t.shape[-1])))==0 and np.max(abs(t[:,0]))==0
            assert all(r['eta_sweeps']==160 and r['corrections']==(20 if method=='projected995' else 0) for r in accepted)
            assert all(r['precision_change']<=1e-10 for r in accepted)
            assert all(r['warp'] for r in accepted[1:])
            assert all(r['cost_after']<=r['cost_before']-1e-4*r['alpha']*r['b_dot_step'] for r in accepted)
            for key,value in data['totals'].items():assert abs(value-sum(r.get(key,0) for r in data['rows']))<1e-9
            with np.load(root/'state.npz') as z:state={k:z[k] for k in z.files}
            L,S,Fi,Fj,offset,raw,_=g.assemble(state['R'],state['t'])
            previous=tuple(state[k] for k in ('p','eta','offset','S','oldR','oldt'))
            pn,en,wm=warp(previous,S,offset,L.directed()[1],state['R'],state['t'])
            RT=state['oldR'].transpose(0,2,1)
            a=log_pose(RT@state['R'],np.einsum('nij,nj->ni',RT,state['t']-state['oldt']))[1:]
            q=np.linalg.solve(state['S'],a[...,None])[...,0];J=np.linalg.solve(state['S'],jr_inv(a)@S)
            dst=L.directed()[1];sel=np.linspace(0,len(dst)-1,min(80,len(dst)),dtype=int)
            rng=np.random.default_rng(211);z=rng.normal(size=(len(sel),L.d))*.03
            qe=q[dst[sel]];x=qe+np.einsum('nij,nj->ni',J[dst[sel]],z)
            po=state['p'][sel];eo=(state['eta']+state['offset'])[sel]
            old=.5*np.einsum('ni,nij,nj->n',x,po,x)-np.sum(eo*x,axis=1)
            constant=.5*np.einsum('ni,nij,nj->n',qe,po,qe)-np.sum(eo*qe,axis=1)
            new=.5*np.einsum('ni,nij,nj->n',z,pn[sel],z)-np.sum((en+offset)[sel]*z,axis=1)
            affine_error=float(np.max(abs(old-constant-new)/np.maximum(1.,abs(old)+abs(constant)+abs(new))))
            assert affine_error<1e-10
            dx=np.zeros((g.n,g.d));dx[1:]=np.einsum('nij,nj->ni',state['S'],state['x'])
            rn,tn=retract(state['oldR'],state['oldt'],dx,accepted[-1]['alpha'])
            pose_error=max(float(np.max(abs(rn-state['R']))),float(np.max(abs(tn-state['t']))))
            assert pose_error==0
            checks[f'{name}_{method}']=dict(accepted_steps=len(accepted),cost_max_relative_error=error,
                canonical_warp_potential_relative_error=affine_error,last_pose_update_error=pose_error,
                final_cost=float(costs[-1]),work_ledger_pass=True)
    return checks


if __name__=='__main__':print(json.dumps(verify(),indent=2))
