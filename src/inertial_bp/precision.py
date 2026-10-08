"""Cancellation-free PSD precision preparation; eta stays unchanged by default."""
import time
import numpy as np
from numba import njit, prange
from .gbp import chol_solve, frozen_sweep
from .diagnostics import build_lift, precision_diagnostics, relative_residual

# Match the original nonlinear engine: precision JIT functions deliberately
# avoid disk caching. Cached nested precision calls caused an access violation
# on process restart in the verified Windows/Numba 0.59.1 environment.

@njit(cache=False)
def step(p,src,dst,rev,As,B,U,u,Fiti,Fi,Fj,rw,ptr,inc):
    ne,d,_=p.shape;P=U.copy();pn=np.empty_like(p);G=np.empty_like(p);const=np.empty((ne,d));change=0.
    for e in range(ne):P[dst[e]]+=p[e]
    for e in range(ne):
        s=src[e];K=U[s].copy()
        for k in range(ptr[s],ptr[s+1]):
            q=inc[k]
            if q!=rev[e]:K+=p[q]
        X,ok=chol_solve(As[e]+K,B[e])
        if not ok:return p,P,G,const,np.inf,e
        E=Fiti[e]@K@X
        pn[e]=E.T@E+X.T@K@X;pn[e]=.5*(pn[e]+pn[e].T);G[e]=-X.T
        const[e]=-E.T@rw[e]+G[e]@u[s]
        scale=max(np.max(np.abs(pn[e])),np.max(np.abs(p[e])),1e-30)
        change=max(change,np.max(np.abs(pn[e]-p[e]))/scale)
    return pn,P,G,const,change,-1


@njit(cache=False)
def iterate(p,eta,src,dst,rev,As,B,U,u,Fiti,Fi,Fj,rw,ptr,inc,tol,cap,joint,omega,eta_limit=np.inf):
    stable=0;t=np.zeros_like(u);buf=eta.copy();eta_start=eta.copy();rejected=False;eta_peak=np.max(np.abs(eta)) if eta.size else 0.
    for it in range(cap):
        pn,P,G,const,change,bad=step(p,src,dst,rev,As,B,U,u,Fiti,Fi,Fj,rw,ptr,inc)
        if bad>=0:return p,eta,P,G,const,it+1,change,bad,rejected,eta_peak
        if joint and not rejected:
            frozen_sweep(eta,buf,t,src,dst,rev,G,const,omega);eta,buf=buf,eta
            current=np.max(np.abs(eta)) if eta.size else 0.
            eta_peak=max(eta_peak,current)
            if not np.isfinite(eta).all() or current>eta_limit:
                eta=eta_start.copy();rejected=True
        if change<tol:stable+=1
        else:stable=0
        if stable>=4:return p,eta,P,G,const,it+1,change,-1,rejected,eta_peak
        p=(1.-omega)*p+omega*pn
    return p,eta,P,G,const,cap,change,-2,rejected,eta_peak


def prepare(L,Fi,Fj,p0=None,eta0=None,joint=False,tol=1e-11,cap=60000,initial='upper',omega=1.,warmup_guard=1e3):
    """Prepare precision independently by default; never return a poisoned warmup.

    ``joint=True`` remains an explicitly requested heuristic. Its eta state is
    discarded if it exceeds the documented amplitude guard or worsens the
    original normal residual relative to the supplied eta0 at frozen precision.
    Neither guard changes p, A or b. Upper initialization uses actual factor
    endpoint blocks, with no artificial unary precision.
    """
    if initial not in ('zero','upper'):raise ValueError('initial must be zero or upper')
    if not 0.<omega<=1.:raise ValueError('omega must be in (0, 1]')
    if not tol>0 or cap<4:raise ValueError('tol must be positive and cap at least 4')
    if not np.isfinite(warmup_guard) or warmup_guard<=0:raise ValueError('warmup_guard must be finite and positive')
    tic=time.perf_counter();src,dst,rev,As,B,Ad,bs,bd=L.directed();n=L.n
    inc=np.argsort(dst,kind='stable').astype(np.int64);ptr=np.r_[0,np.cumsum(np.bincount(dst,minlength=n))].astype(np.int64)
    ffi=np.ascontiguousarray(np.concatenate([Fi,Fj]));ffj=np.ascontiguousarray(np.concatenate([Fj,Fi]))
    Fiti=np.ascontiguousarray(np.linalg.inv(ffi.transpose(0,2,1)))
    rw=np.linalg.solve(Fi.transpose(0,2,1),-L.bi[...,None])[...,0];rw=np.ascontiguousarray(np.concatenate([rw,rw]))
    p=(Ad.copy() if initial=='upper' else np.zeros_like(Ad)) if p0 is None else p0.copy()
    eta=np.zeros_like(bs) if eta0 is None else eta0.copy()
    eta_start=eta.copy()
    if not np.isfinite(p).all() or not np.isfinite(eta).all():raise ValueError('Nonfinite initial message state')
    limit=warmup_guard*max(float(np.max(np.abs(L.b))),float(np.max(np.abs(eta))) if eta.size else 0.,1e-30)
    p,eta,P,G,const,it,change,bad,rejected,peak=iterate(p,eta,src,dst,rev,As,B,L.U,L.u,Fiti,ffi,ffj,rw,ptr,inc,tol,cap,joint,omega,limit)
    if bad!=-1 or not np.isfinite(change):raise ArithmeticError(f'precision failure {bad} iter={it} change={change}')
    diagnostics=precision_diagnostics(L,p,P,G)
    invP=np.linalg.inv(P)
    lift,mean_map,ranks,lift_diagnostics=build_lift(p,dst,L.U,P)
    diagnostics.update(lift_diagnostics)
    initial_rr=relative_residual(L,eta_start,dst,invP)
    warm_rr=relative_residual(L,eta,dst,invP)
    reason='amplitude_or_nonfinite_guard' if rejected else 'none'
    if joint and (not np.isfinite(warm_rr) or warm_rr>initial_rr*(1.+1e-12)):
        eta=eta_start.copy();rejected=True;reason='normal_residual_guard'
    diagnostics.update(joint_warmup_requested=bool(joint),joint_warmup_rejected=bool(rejected),
        warmup_rejection_reason=reason,warmup_eta_peak=float(peak),warmup_eta_limit=float(limit),
        warmup_relative_residual=float(warm_rr),initial_relative_residual=float(initial_rr),
        returned_relative_residual=relative_residual(L,eta,dst,invP),
        precision_initialization=initial if p0 is None else 'provided',
        eta_initialization='joint_guarded' if joint and not rejected else 'provided_unchanged')
    return dict(src=src,dst=dst,rev=rev,p=p,P=P,invP=invP,G=np.ascontiguousarray(G),const=np.ascontiguousarray(const),
                lift=np.ascontiguousarray(lift),eta0=eta,precision_sweeps=int(it),precision_change=float(change),prepare_seconds=time.perf_counter()-tic,
                ptr=ptr,inc=inc,Fi=ffi,Fj=ffj,rw=rw,lift_mean_map=mean_map,incoming_ranks=ranks,**diagnostics)

serial_precision_step=step

@njit(cache=False,parallel=True)
def fused_precision_step(p,src,dst,rev,As,B,U,u,Fiti,Fi,Fj,rw,ptr,inc):
    """Same cavity/Cholesky/PSD formula, explicit 3x3/6x6 scalar products.

    A single scratch allocation avoids tiny BLAS calls and parallel allocator
    contention. No fastmath, precision approximation, or altered damping.
    """
    ne,d,_=p.shape
    P=U.copy();pn=np.empty_like(p);G=np.empty_like(p);const=np.empty((ne,d))
    scratch=np.empty((ne,6,d,d));changes=np.zeros(ne);bad=np.zeros(ne,np.int64)
    for e in range(ne):
        for a in range(d):
            for b in range(d):P[dst[e],a,b]+=p[e,a,b]
    for e in prange(ne):
        s=src[e];K=scratch[e,0];L=scratch[e,1];Y=scratch[e,2];X=scratch[e,3];Z=scratch[e,4];E=scratch[e,5]
        for a in range(d):
            for b in range(d):K[a,b]=U[s,a,b]
        for k in range(ptr[s],ptr[s+1]):
            q=inc[k]
            if q!=rev[e]:
                for a in range(d):
                    for b in range(d):K[a,b]+=p[q,a,b]
        for a in range(d):
            for b in range(a+1):
                v=As[e,a,b]+K[a,b]
                for c in range(b):v-=L[a,c]*L[b,c]
                if a==b:
                    if v<=0. or not np.isfinite(v):bad[e]=1;break
                    L[a,b]=np.sqrt(v)
                else:L[a,b]=v/L[b,b]
            if bad[e]:break
        if bad[e]:continue
        for b in range(d):
            for a in range(d):
                v=B[e,a,b]
                for c in range(a):v-=L[a,c]*Y[c,b]
                Y[a,b]=v/L[a,a]
            for ar in range(d):
                a=d-1-ar;v=Y[a,b]
                for c in range(a+1,d):v-=L[c,a]*X[c,b]
                X[a,b]=v/L[a,a]
        for a in range(d):
            for b in range(d):
                v=0.
                for c in range(d):v+=Fiti[e,a,c]*K[c,b]
                Z[a,b]=v
        for a in range(d):
            for b in range(d):
                v=0.
                for c in range(d):v+=Z[a,c]*X[c,b]
                E[a,b]=v
        for a in range(d):
            for b in range(d):
                v=0.
                for c in range(d):v+=X[c,a]*K[c,b]
                Z[a,b]=v
        for a in range(d):
            for b in range(d):
                v=0.;w=0.
                for c in range(d):v+=E[c,a]*E[c,b];w+=Z[a,c]*X[c,b]
                pn[e,a,b]=v+w;G[e,a,b]=-X[b,a]
        for a in range(d):
            for b in range(a,d):
                v=.5*(pn[e,a,b]+pn[e,b,a]);pn[e,a,b]=v;pn[e,b,a]=v
        scale=1e-30;change=0.
        for a in range(d):
            v=0.;w=0.
            for c in range(d):v-=E[c,a]*rw[e,c];w+=G[e,a,c]*u[s,c]
            const[e,a]=v+w
            for b in range(d):
                scale=max(scale,abs(pn[e,a,b]),abs(p[e,a,b]))
                change=max(change,abs(pn[e,a,b]-p[e,a,b]))
        changes[e]=change/scale
    for e in range(ne):
        if bad[e]:return p,P,G,const,np.inf,e
    return pn,P,G,const,np.max(changes),-1

step=fused_precision_step
