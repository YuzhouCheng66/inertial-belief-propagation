import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('NUMBA_NUM_THREADS','8')
import unittest
from pathlib import Path
import numpy as np
from inertial_bp.benchmarks import load_frozen
from inertial_bp.solver import solve, inject
from inertial_bp.nonlinear import load_case
from inertial_bp.models import log_pose,jr_inv
from inertial_bp.lie import retract
from inertial_bp.warp import warp
from inertial_bp.precision import prepare
from inertial_bp.preconditioner import build_local_pc

ROOT=Path(__file__).resolve().parents[1]


class AlgorithmTests(unittest.TestCase):
    def test_original_frozen_prefix_and_independent_beta(self):
        for name in ('smallGrid3D','FR079','Sphere','Cubicle'):
            with self.subTest(dataset=name):
                L,P,pc,offset=load_frozen(name)
                out=solve(L,P,pc,offset,sweeps=160,tol=0.)
                with np.load(ROOT/f'results/frozen/{name}_ibp.npz') as z:expected=z['trace'][:20]
                # Final historical checkpoint intentionally has no correction.
                np.testing.assert_allclose(out['trace'][:-1],expected[:-1],rtol=2e-11,atol=2e-9)
                np.testing.assert_allclose(out['trace'][-1,:8],expected[-1,:8],rtol=2e-11,atol=2e-9)
                self.assertEqual(out['metrics']['corrections'],19)
                self.assertEqual(out['metrics']['F_calls'],180)
                self.assertEqual(out['beta_history'].shape,(20,(L.n+31)//32))
                # Identical local clocks can coincide before any local restart.
                if name in ('FR079','Cubicle'):
                    self.assertTrue(np.any(np.ptp(out['beta_history'],axis=1)>0))
                self.assertLessEqual(out['beta_history'].max(),.995)

    def test_pure_gbp_is_actual_message_iteration(self):
        with np.load(ROOT/'results/frozen/comparison.npz') as z:
            for name in ('smallGrid3D','FR079','Sphere','Cubicle'):
                L,P,pc,offset=load_frozen(name)
                out=solve(L,P,pc,offset,method='gbp',sweeps=160,tol=0.)
                expected=z[f'{name}_pure'][:20,:5]
                np.testing.assert_allclose(out['trace'][:,:5],expected,rtol=2e-11,atol=2e-9)
                self.assertEqual(out['metrics']['F_calls'],161)
                self.assertEqual(out['metrics']['corrections'],0)

    def test_complete_20_cycles_match_next_original_prefix(self):
        for name in ('FR079','Cubicle'):
            L,P,pc,offset=load_frozen(name)
            out=solve(L,P,pc,offset,sweeps=160,tol=0.,complete_cycles=True)
            with np.load(ROOT/f'results/frozen/{name}_ibp.npz') as z:expected=z['trace'][:20]
            np.testing.assert_allclose(out['trace'],expected,rtol=2e-11,atol=2e-9)
            self.assertEqual(out['metrics']['corrections'],20)
            self.assertEqual(out['metrics']['sweeps'],160)
            self.assertEqual(out['metrics']['F_calls'],181)

    def test_eta_lift_achieves_supported_mean_correction(self):
        L,P,pc,offset=load_frozen('FR079')
        rng=np.random.default_rng(139);delta=rng.normal(size=L.b.shape)
        eta=np.zeros_like(offset);inject(eta,delta,P['src'],P['dst'],P['lift'])
        total=np.zeros_like(delta);np.add.at(total,P['dst'],eta)
        achieved=np.einsum('nij,nj->ni',P['invP'],total)
        np.testing.assert_allclose(achieved,delta,rtol=1e-9,atol=1e-9)

    def test_lie_chart_precision_eta_and_covariance_warp(self):
        rng=np.random.default_rng(411)
        for k,d in ((2,3),(3,6)):
            n=5;oldR=np.repeat(np.eye(k)[None],n,axis=0);oldt=np.zeros((n,k))
            shift=rng.normal(size=(n,d))*.05;shift[0]=0
            R,t=retract(oldR,oldt,shift)
            def spd():
                a=rng.normal(size=(n-1,d,d));return a.transpose(0,2,1)@a+np.eye(d)
            oldS=spd();S=spd();p=spd();dst=np.arange(n-1)
            eta=rng.normal(size=(n-1,d));oldoff=rng.normal(size=eta.shape);offset=rng.normal(size=eta.shape)
            pp,en,_=warp((p,eta,oldoff,oldS,oldR,oldt),S,offset,dst,R,t)
            a=log_pose(oldR.transpose(0,2,1)@R,t-oldt)[1:]
            q=np.linalg.solve(oldS,a[...,None])[...,0];J=np.linalg.solve(oldS,jr_inv(a)@S)
            z=rng.normal(size=eta.shape)*.1;x=q+np.einsum('nij,nj->ni',J,z)
            old=.5*np.einsum('ni,nij,nj->n',x,p,x)-np.sum((eta+oldoff)*x,axis=1)
            constant=.5*np.einsum('ni,nij,nj->n',q,p,q)-np.sum((eta+oldoff)*q,axis=1)
            new=.5*np.einsum('ni,nij,nj->n',z,pp,z)-np.sum((en+offset)*z,axis=1)
            np.testing.assert_allclose(old-constant,new,rtol=2e-10,atol=2e-10)
            Ji=np.linalg.inv(J)
            np.testing.assert_allclose(np.linalg.inv(pp),Ji@np.linalg.inv(p)@Ji.transpose(0,2,1),rtol=1e-10,atol=1e-10)
            direction=rng.normal(size=(n,d))*.03;direction[0]=0;eps=1e-6
            rn,tn=retract(R,t,direction,eps)
            an=log_pose(oldR.transpose(0,2,1)@rn,tn-oldt)[1:]
            np.testing.assert_allclose((an-a)/eps,np.einsum('nij,nj->ni',jr_inv(a),direction[1:]),rtol=3e-6,atol=1e-8)

    def test_fresh_raw_input_preparation_and_local_basis(self):
        for name in ('smallGrid3D','FR079'):
            g=load_case(name)
            L,S,Fi,Fj,offset,raw,_=g.assemble(g.R0,g.t0)
            cached,_,_,off=load_frozen(name)
            for key in ('D','Hij','b'):
                np.testing.assert_allclose(getattr(L,key),getattr(cached,key),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(offset,off,rtol=1e-12,atol=1e-12)
            P=prepare(L,Fi,Fj,eta0=-offset,tol=1e-10,cap=20000,omega=.5)
            pc,_,_=build_local_pc(L,raw,S,group=32)
            out=solve(L,P,pc,offset,eta=P['eta0'],sweeps=160,tol=0.,complete_cycles=True)
            with np.load(ROOT/f'results/frozen/{name}_ibp.npz') as z:expected=z['trace'][:20]
            np.testing.assert_allclose(out['trace'][:,:7],expected[:,:7],rtol=1e-7,atol=2e-7)
            self.assertLessEqual(P['precision_change'],1e-10)

    def test_invalid_run_parameters(self):
        L,P,pc,offset=load_frozen('FR079')
        for kw in ({'sweeps':7},{'sweeps':17},{'tol':-1},{'method':'shared-beta'}):
            with self.assertRaises(ValueError):solve(L,P,pc,offset,**kw)


if __name__=='__main__':unittest.main()
