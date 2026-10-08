"""Affine pullback of precision and canonical eta across Lie charts."""
import time
import numpy as np
from .models import log_pose, jr_inv

def warp(previous,S,offset,dst,R,t):
    """Same affine Gaussian pullback as HGBP warm-transport, plus whitening."""
    start=time.perf_counter()
    p,eta,oldoff,oldS,oldR,oldt=previous
    RT=oldR.transpose(0,2,1)
    a=log_pose(RT@R,np.einsum('nij,nj->ni',RT,t-oldt))[1:]
    M=jr_inv(a)
    q=np.linalg.solve(oldS,a[...,None])[...,0]
    J=np.linalg.solve(oldS,M@S)
    Je=J[dst];qe=q[dst]
    pp=Je.transpose(0,2,1)@p@Je
    pp=.5*(pp+pp.transpose(0,2,1))
    en=np.einsum('eij,ej->ei',Je.transpose(0,2,1),eta+oldoff-np.einsum('eij,ej->ei',p,qe))-offset
    ev,V=np.linalg.eigh(pp)
    mask=ev[:,0]<0
    projection=0.
    if mask.any():
        fixed=(V[mask]*np.maximum(ev[mask],0)[:,None,:])@V[mask].transpose(0,2,1)
        projection=float(np.linalg.norm(fixed-pp[mask])/max(np.linalg.norm(pp[mask]),1e-300))
        pp[mask]=fixed
    assert np.isfinite(pp).all() and np.isfinite(en).all()
    return pp,en,dict(seconds=time.perf_counter()-start,max_chart_condition=float(np.max(np.linalg.cond(J))),
        projected_slots=int(mask.sum()),relative_psd_projection=projection,max_rotation_shift=float(np.max(np.linalg.norm(a[:,2:] if a.shape[1]==3 else a[:,3:],axis=1))))
