"""Actual synchronous frozen-precision Gaussian BP message sweeps."""
import numpy as np
from numba import njit

@njit(cache=True)
def chol_solve(A,B):
    d=A.shape[0];L=np.zeros((d,d));X=np.empty_like(B);Y=np.empty_like(B)
    for i in range(d):
        for j in range(i+1):
            v=A[i,j]
            for k in range(j):v-=L[i,k]*L[j,k]
            if i==j:
                if v<=0. or not np.isfinite(v):return X,False
                L[i,j]=np.sqrt(v)
            else:L[i,j]=v/L[j,j]
    for c in range(B.shape[1]):
        for i in range(d):
            v=B[i,c]
            for k in range(i):v-=L[i,k]*Y[k,c]
            Y[i,c]=v/L[i,i]
        for ii in range(d):
            i=d-1-ii;v=Y[i,c]
            for k in range(i+1,d):v-=L[k,i]*X[k,c]
            X[i,c]=v/L[i,i]
    return X,True


@njit(cache=True)
def sums(eta,dst,u,t):
    n,d=u.shape
    for i in range(n):
        for a in range(d):t[i,a]=u[i,a]
    for e in range(len(dst)):
        for a in range(d):t[dst[e],a]+=eta[e,a]


@njit(cache=True)
def frozen_sweep(eta,out,t,src,dst,rev,G,const,omega=1.):
    t[:]=0.;d=eta.shape[1];v=np.empty(d)
    for e in range(len(dst)):
        for a in range(d):t[dst[e],a]+=eta[e,a]
    for e in range(len(src)):
        for a in range(d):v[a]=t[src[e],a]-eta[rev[e],a]
        for a in range(d):
            z=const[e,a]
            for b in range(d):z+=G[e,a,b]*v[b]
            if G.shape[2]==2*d:
                for b in range(d):z+=G[e,a,d+b]*t[dst[e],b]
            if omega==1.:out[e,a]=z
            else:out[e,a]=(1.-omega)*eta[e,a]+omega*z


@njit(cache=True)
def means(eta,dst,u,invP,t,x):
    sums(eta,dst,u,t);n,d=x.shape
    for i in range(n):
        for a in range(d):
            z=0.
            for b in range(d):z+=invP[i,a,b]*t[i,b]
            x[i,a]=z


@njit(cache=True)
def residual(x,i,j,D,Hij,b,h):
    n,d=x.shape
    for k in range(n):
        for a in range(d):
            z=b[k,a]
            for c in range(d):z-=D[k,a,c]*x[k,c]
            h[k,a]=z
    for e in range(len(i)):
        for a in range(d):
            for c in range(d):
                h[i[e],a]-=Hij[e,a,c]*x[j[e],c]
                h[j[e],c]-=Hij[e,a,c]*x[i[e],a]
