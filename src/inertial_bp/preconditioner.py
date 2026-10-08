"""Balanced block-local residual correction and local physical low modes."""
import time
import numpy as np
from scipy.linalg import eigh
from numba import njit

def build_local_pc(L,raw,S,cache=None,refresh=True,group=20,rank=None):
    """Return (phase2-compatible pc tuple, physical basis cache, metadata).

    Group membership and endpoint ordering must match between L and raw.
    Reuse identifies the right-trivialized Lie-algebra coordinates between
    poses; it does not claim to preserve a fixed world-space displacement.
    The .9 global upper bound requires the audited PSD pairwise graph model.
    """
    begin=time.perf_counter();n,d=L.n,L.d
    rank=(4 if d==3 else 12) if rank is None else rank
    if not isinstance(group,(int,np.integer)) or group<=0:raise ValueError('group must be a positive integer')
    if not isinstance(rank,(int,np.integer)) or rank<=0:raise ValueError('rank must be a positive integer')
    if n<=0 or raw.n!=n or raw.d!=d or np.shape(S)!=(n,d,d):raise ValueError('incompatible linear graph dimensions')
    if not np.array_equal(L.i,raw.i) or not np.array_equal(L.j,raw.j):raise ValueError('normalized/raw endpoints differ')
    ng=(n+group-1)//group;m=group*d
    if not refresh:
        if cache is None:raise ValueError('basis reuse requires a physical basis cache')
        if any(cache.get(k)!=v for k,v in dict(n=n,d=d,group=group,rank=rank).items()):raise ValueError('incompatible physical basis cache')
        if np.shape(cache.get('physical_basis'))!=(ng,m,rank):raise ValueError('physical basis cache shape mismatch')
        if not np.isfinite(cache['physical_basis']).all():raise ValueError('nonfinite physical basis cache')
    start=time.perf_counter();invD=np.linalg.inv(L.D);base=.225*invD
    U=np.zeros((ng,m,rank));X=np.zeros_like(U);V=np.zeros_like(U);K=np.zeros((ng,rank,rank))
    physical=np.zeros_like(U) if refresh else cache['physical_basis'].copy()
    internal=[[] for _ in range(ng)]
    for e,(i,j) in enumerate(zip(L.i,L.j)):
        gi=int(i)//group
        if gi==int(j)//group:internal[gi].append(e)
    classify_seconds=time.perf_counter()-start
    assemble_seconds=0.;basis_seconds=0.;curvature_seconds=0.;diag=[]
    for g in range(ng):
        start=time.perf_counter();lo=g*group;hi=min(lo+group,n);nn=hi-lo;z=nn*d;rr=min(rank,z)
        Hg=np.zeros((z,z));Ng=np.zeros((z,z)) if refresh else None
        M=np.zeros_like(Hg)
        for a in range(nn):
            sl=slice(a*d,(a+1)*d);Hg[sl,sl]=L.D[lo+a];M[sl,sl]=.5*invD[lo+a]
            if refresh:Ng[sl,sl]=raw.U[lo+a]
        for e in internal[g]:
            i=int(L.i[e])-lo;j=int(L.j[e])-lo;si=slice(i*d,(i+1)*d);sj=slice(j*d,(j+1)*d)
            Hg[si,sj]+=L.Hij[e];Hg[sj,si]+=L.Hij[e].T
            if refresh:
                Ng[si,si]+=raw.Hii[e];Ng[sj,sj]+=raw.Hjj[e]
                Ng[si,sj]+=raw.Hij[e];Ng[sj,si]+=raw.Hij[e].T
        assemble_seconds+=time.perf_counter()-start
        start=time.perf_counter()
        if refresh:
            ev,Phi=eigh(Ng,subset_by_index=[0,rr-1],check_finite=False)
            physical[g,:z,:rr]=Phi
        else:ev=None;Phi=physical[g,:z,:rr]
        mapped=np.linalg.solve(S[lo:hi],Phi.reshape(nn,d,rr)).reshape(z,rr)
        Z,R=np.linalg.qr(mapped,mode='reduced')
        if not np.isfinite(Z).all() or np.linalg.matrix_rank(R)<rr:raise ValueError('physical basis lost rank after coordinate mapping')
        V[g,:z,:rr]=Z;basis_seconds+=time.perf_counter()-start
        start=time.perf_counter();E=Z.T@Hg@Z;ee=np.linalg.eigvalsh(E)
        if ee[0]<=0:raise ValueError('non-SPD current projected local Hessian')
        C=np.linalg.cholesky(E);u=np.linalg.solve(C,Z.T).T
        w=Hg@u;x=M@w;kk=np.eye(rr)+w.T@x
        U[g,:z,:rr]=u;X[g,:z,:rr]=x;K[g,:rr,:rr]=kk
        diag.append(dict(group=g,min_mode=None if ev is None else float(ev[0]),
                         max_retained=None if ev is None else float(ev[-1]),
                         projected_condition=float(ee[-1]/ee[0]),
                         local_analytic_upper_bound=.45,full_spectrum_diagnostic_computed=False))
        curvature_seconds+=time.perf_counter()-start
    elapsed=time.perf_counter()-begin
    newcache=dict(n=n,d=d,group=group,rank=rank,physical_basis=physical,
                  coordinate_identification='right-Lie components; identity between linearizations')
    metadata=dict(setup_seconds=elapsed,refresh=bool(refresh),groups=ng,rank=rank,
                  internal_edges=sum(map(len,internal)),input_edges=len(L.i),
                  classify_and_allocate_seconds=classify_seconds,assemble_local_seconds=assemble_seconds,
                  basis_eigen_map_qr_seconds=basis_seconds,current_curvature_seconds=curvature_seconds,
                  local_analytic_upper_bound=.45,global_analytic_upper_bound=.9,
                  bound_scope='exact arithmetic, current PSD pairwise model and full-rank cached basis',
                  reused_old_preconditioner=False,full_group_spectral_diagnostics=False,
                  basis_coordinate_identification=newcache['coordinate_identification'])
    return (base,U,X,K,V,diag,elapsed),newcache,metadata

@njit(cache=True)
def _step(h,base,U,X,K,group):
 v=np.empty_like(h)
 for i in range(len(h)):v[i]=base[i]@h[i]
 for g in range(len(U)):
  lo=g*group;hi=min(len(h),lo+group);z=(hi-lo)*h.shape[1]
  if U.shape[2]:
   u=U[g,:z];xx=X[g,:z];hh=h[lo:hi].reshape(z);q=u.T@hh
   v[lo:hi]+=(.45*(u@(K[g]@q-xx.T@hh)-xx@q)).reshape(hi-lo,h.shape[1])
 return v
