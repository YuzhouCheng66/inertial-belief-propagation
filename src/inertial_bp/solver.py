"""Independent group beta, no filters, eight GBP sweeps per correction."""
import time
import numpy as np
from numba import njit
from .gbp import frozen_sweep, means, residual
from .preconditioner import _step

@njit(cache=True)
def project(v,V,group):
    a=np.zeros((len(V),V.shape[2])); n,d=v.shape
    for g in range(len(V)):
        lo=g*group; hi=min(n,lo+group); z=(hi-lo)*d
        a[g]=V[g,:z].T@v[lo:hi].reshape(z)
    return a


@njit(cache=True)
def expand(a,V,n,d,group):
    v=np.zeros((n,d))
    for g in range(len(V)):
        lo=g*group;hi=min(n,lo+group);z=(hi-lo)*d
        v[lo:hi]=(V[g,:z]@a[g]).reshape(hi-lo,d)
    return v


@njit(cache=True)
def inject(eta,v,src,dst,lift):
    d=v.shape[1]
    for e in range(len(eta)):
        eta[e]+=lift[e,:,:d]@v[dst[e]]
        if lift.shape[2]==2*d:eta[e]+=lift[e,:,d:]@v[src[e]]


TRACE_COLUMNS = (
    'sweeps', 'kernel_F_calls', 'normal_residual', 'message_residual',
    'quadratic_energy', 'mean_beta', 'max_beta', 'cumulative_restarts',
    'correction_energy_change', 'post_correction_normal_residual',
    'reserved_10', 'reserved_11', 'shifted_eta_norm', 'desired_correction_norm',
    'reserved_14', 'reserved_15', 'reserved_16', 'reserved_17',
)


@njit(cache=True)
def _run(eta,offset,src,dst,rev,G,const,invP,u,lift,ii,jj,D,Hij,b,
         base,U,X,K,V,mode,cap,tol,complete):
    n,d=b.shape;ng=len(V)
    tmp=np.zeros_like(b);x=tmp.copy();h=tmp.copy();prev=tmp.copy()
    beta=np.zeros(ng);clock=np.ones(ng)
    buf=eta.copy();prop=eta.copy()
    bn=max(np.sqrt(np.sum(b*b)),1e-300);cycles=cap//8
    trace=np.zeros((cycles,18));beta_history=np.zeros((cycles,ng))
    sw=0;fc=0;corrections=0;resets=0;status=0;count=0
    for cy in range(cycles):
        # Reuse the GBP diagnostic proposal as the next accepted sweep.
        first=1 if mode==0 and cy>0 else 0
        for s in range(first,8):
            frozen_sweep(eta,buf,tmp,src,dst,rev,G,const,1.);eta,buf=buf,eta;fc+=1
        sw+=8
        frozen_sweep(eta,prop,tmp,src,dst,rev,G,const,1.);fc+=1
        means(eta,dst,u,invP,tmp,x);residual(x,ii,jj,D,Hij,b,h)
        rr=np.sqrt(np.sum(h*h))/bn
        mr=np.max(np.abs(prop-eta))/(1.+np.max(np.abs(eta+offset)))
        energy=-.5*np.sum(x*(b+h))
        trace[cy,0]=sw;trace[cy,1]=fc;trace[cy,2]=rr;trace[cy,3]=mr;trace[cy,4]=energy
        trace[cy,5]=np.mean(beta);trace[cy,6]=np.max(beta);trace[cy,7]=resets
        beta_history[cy]=beta;count=cy+1
        if not np.isfinite(rr+mr) or rr>1e10:status=2;break
        if rr<=tol and mr<=tol:status=1;break
        if cy==cycles-1 and (not complete or mode==0):break
        if mode==0:
            eta,prop=prop,eta
            continue
        old_x=x.copy();old_h=h.copy();dd=x-prev if cy>0 else np.zeros_like(x)
        momentum=expand(project(dd,V,32),V,n,d,32)
        for g in range(ng):momentum[g*32:min(n,(g+1)*32)]*=beta[g]
        desired=_step(h,base,U,X,K,32)
        desired+=momentum
        inject(eta,desired,src,dst,lift)
        # Store the pre-correction checkpoint; each group updates only its own clock.
        prev[:]=old_x
        for g in range(ng):
            lo=g*32;hi=min(n,lo+32)
            dot=np.sum(old_h[lo:hi]*(dd[lo:hi]+desired[lo:hi])) if cy>0 else 0.
            if dot<0:clock[g]=1.;beta[g]=0.;resets+=1
            else:
                q=.5*(1.+np.sqrt(1.+4.*clock[g]*clock[g]))
                beta[g]=min(.995,(clock[g]-1.)/q);clock[g]=q
        means(eta,dst,u,invP,tmp,x);residual(x,ii,jj,D,Hij,b,h)
        trace[cy,8]=-.5*np.sum((x-old_x)*(old_h+h))
        trace[cy,9]=np.sqrt(np.sum(h*h))/bn
        trace[cy,12]=np.sqrt(np.sum(eta*eta));trace[cy,13]=np.sqrt(np.sum(desired*desired))
        corrections+=1
    return eta,x,trace[:count],beta_history[:count],sw,fc,corrections,resets,status


def empty_pc(L):
    ng=(L.n+31)//32;rank=4 if L.d==3 else 12
    return (np.zeros_like(L.D),np.zeros((ng,32*L.d,rank)),
            np.zeros((ng,32*L.d,rank)),np.zeros((ng,rank,rank)),
            np.zeros((ng,32*L.d,rank)))


def solve(L,P,pc,offset,*,eta=None,method='ibp',sweeps=8192,tol=1e-6,complete_cycles=False):
    """Run original 8+1 with independent group beta (or the pure GBP comparator).

    Frozen convergence trajectories stop before the final correction. With
    ``complete_cycles=True`` the last IBP correction is applied; nonlinear
    experiments use 160 sweeps, tol=0, hence 20 complete cycles.
    ``eta`` is in shifted canonical coordinates; None means canonical zero.
    The algorithm's grouping is fixed at 32 and beta cap at .995.
    """
    if method not in ('ibp','gbp'):raise ValueError('method must be ibp or gbp')
    if not isinstance(sweeps,(int,np.integer)) or sweeps<8 or sweeps%8:
        raise ValueError('sweeps must be a positive multiple of eight')
    if not np.isfinite(tol) or tol<0:raise ValueError('tol must be finite and nonnegative')
    if L.d not in (3,6) or L.n<1:raise ValueError('expected nonempty SE2/SE3 node blocks')
    if method=='gbp':pc=empty_pc(L)
    if pc is None:raise ValueError('IBP needs a local preconditioner')
    if pc[4].shape[0]!=(L.n+31)//32 or pc[4].shape[1]!=32*L.d:
        raise ValueError('preconditioner must use contiguous 32-node groups')
    initial=-offset.copy() if eta is None else np.asarray(eta).copy()
    if initial.shape!=offset.shape or not np.isfinite(initial).all():
        raise ValueError('invalid initial eta')
    start=time.perf_counter()
    eta,x,tr,bh,sw,fc,co,restarts,status=_run(initial,offset,
        *[P[k] for k in ('src','dst','rev','G','const','invP')],L.u,P['lift'],
        L.i,L.j,L.D,L.Hij,L.b,*pc[:5],1 if method=='ibp' else 0,sweeps,tol,complete_cycles)
    normal=float(tr[-1,2]);message=float(tr[-1,3]);extra=0
    if complete_cycles:
        tmp=np.zeros_like(x);h=tmp.copy();proposal=np.empty_like(eta)
        residual(x,L.i,L.j,L.D,L.Hij,L.b,h)
        frozen_sweep(eta,proposal,tmp,P['src'],P['dst'],P['rev'],P['G'],P['const'],1.)
        normal=float(np.linalg.norm(h)/max(np.linalg.norm(L.b),1e-300))
        message=float(np.max(abs(proposal-eta))/(1+np.max(abs(eta+offset))));extra=1
    metrics=dict(method=method,sweeps=int(sw),F_calls=int(fc)+extra,kernel_F_calls=int(fc),
        final_diagnostic_F_calls=extra,corrections=int(co),filter_rounds=0,restarts=int(restarts),
        status=('cap','converged','diverged')[status],normal_residual=normal,message_residual=message,
        seconds=time.perf_counter()-start,mean_beta=float(np.mean(tr[:,5])),
        max_beta=float(np.max(tr[:,6])),complete_cycles=bool(complete_cycles),
        direction_global_reductions=0)
    return dict(eta=eta,x=x,trace=tr,beta_history=bh,metrics=metrics)
