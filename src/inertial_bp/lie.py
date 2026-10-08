"""Right Lie charts and audited pose-graph input parsers."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation

def skew(v):
    v=np.asarray(v);out=np.zeros(v.shape[:-1]+(3,3))
    out[...,0,1]=-v[...,2];out[...,0,2]=v[...,1]
    out[...,1,0]=v[...,2];out[...,1,2]=-v[...,0]
    out[...,2,0]=-v[...,1];out[...,2,1]=v[...,0]
    return out


def rot2(a):
    a=np.asarray(a);c=np.cos(a);s=np.sin(a);out=np.empty(a.shape+(2,2))
    out[...,0,0]=c;out[...,0,1]=-s;out[...,1,0]=s;out[...,1,1]=c
    return out


def retract(R,t,dx,scale=1.):
    """T_new = T Exp(scale*dx^), xi=(rho,phi), true SE exponential."""
    z=np.asarray(dx)*scale;k=R.shape[1]
    if k==2:
        a=z[:,2];a2=a*a;small=np.abs(a)<1e-5
        sa=np.empty_like(a);ca=np.empty_like(a)
        sa[small]=1-a2[small]/6+a2[small]**2/120
        ca[small]=a[small]/2-a[small]*a2[small]/24+a[small]*a2[small]**2/720
        sa[~small]=np.sin(a[~small])/a[~small]
        ca[~small]=(1-np.cos(a[~small]))/a[~small]
        v=np.c_[sa*z[:,0]-ca*z[:,1],ca*z[:,0]+sa*z[:,1]]
        return R@rot2(a),t+np.einsum('nij,nj->ni',R,v)
    phi=z[:,3:];th2=np.einsum('ni,ni->n',phi,phi);th=np.sqrt(th2)
    small=th<1e-4;A=np.empty_like(th);B=A.copy();C=A.copy()
    A[small]=1-th2[small]/6+th2[small]**2/120
    B[small]=.5-th2[small]/24+th2[small]**2/720
    C[small]=1/6-th2[small]/120+th2[small]**2/5040
    A[~small]=np.sin(th[~small])/th[~small]
    B[~small]=(1-np.cos(th[~small]))/th2[~small]
    C[~small]=(th[~small]-np.sin(th[~small]))/(th[~small]*th2[~small])
    K=skew(phi);KK=K@K;I=np.eye(3)
    eR=I+A[:,None,None]*K+B[:,None,None]*KK
    V=I+B[:,None,None]*K+C[:,None,None]*KK
    v=np.einsum('nij,nj->ni',V,z[:,:3])
    return R@eR,t+np.einsum('nij,nj->ni',R,v)


def residual_jacobian(Ri,ti,Rj,tj,Rz,tz):
    """Exact analytic first derivatives with respect to right Lie increments."""
    n=len(ti);k=Ri.shape[-1];d=3 if k==2 else 6
    RiT=Ri.transpose(0,2,1);RzT=Rz.transpose(0,2,1)
    Rrel=RiT@Rj;trel=np.einsum('nij,nj->ni',RiT,tj-ti)
    Re=RzT@Rrel;te=np.einsum('nij,nj->ni',RzT,trel-tz)
    e=np.empty((n,d));e[:,:k]=te
    Ji=np.zeros((n,d,d));Jj=Ji.copy()
    Ji[:,:k,:k]=-RzT;Jj[:,:k,:k]=Re
    if k==2:
        e[:,2]=np.arctan2(Re[:,1,0],Re[:,0,0])
        Ji[:,:2,2]=np.einsum('nij,nj->ni',RzT,np.c_[trel[:,1],-trel[:,0]])
        Ji[:,2,2]=-1;Jj[:,2,2]=1
    else:
        q=Rotation.from_matrix(Re).as_quat()
        q[q[:,3]<0]*=-1
        v=q[:,:3];w=q[:,3];e[:,3:]=v
        Ji[:,:3,3:]=RzT@skew(trel)
        Ji[:,3:,3:]=-.5*(w[:,None,None]*np.eye(3)-skew(v))@RzT
        Jj[:,3:,3:]=.5*(w[:,None,None]*np.eye(3)+skew(v))
    return e,Ji,Jj


@dataclass
class PoseGraph:
    name:str
    i:np.ndarray
    j:np.ndarray
    Rz:np.ndarray
    tz:np.ndarray
    Omega:np.ndarray
    R0:np.ndarray
    t0:np.ndarray
    anchor_weight:float=1000.
    source:dict|None=None
    @property
    def n(self):return len(self.t0)
    @property
    def d(self):return 3 if self.R0.shape[-1]==2 else 6
    def evaluate(self,R,t,jac=False):
        e,Ji,Jj=residual_jacobian(R[self.i],t[self.i],R[self.j],t[self.j],self.Rz,self.tz)
        k=t.shape[1]
        a,_,Ja=residual_jacobian(np.eye(k)[None],np.zeros((1,k)),R[:1],t[:1],self.R0[:1],self.t0[:1])
        cost=.5*np.einsum('ei,eij,ej->',e,self.Omega,e)+.5*self.anchor_weight*np.sum(a*a)
        if jac:return cost,e,Ji,Jj,a[0],Ja[0]
        return float(cost)
    def linearize(self,R,t):
        cost,e,Ji,Jj,a,Ja=self.evaluate(R,t,True)
        OJi=self.Omega@Ji;OJj=self.Omega@Jj
        Hii=Ji.transpose(0,2,1)@OJi;Hij=Ji.transpose(0,2,1)@OJj;Hjj=Jj.transpose(0,2,1)@OJj
        oe=np.einsum('eab,eb->ea',self.Omega,e)
        bi=-np.einsum('eab,ea->eb',Ji,oe);bj=-np.einsum('eab,ea->eb',Jj,oe)
        U=np.zeros((self.n,self.d,self.d));u=np.zeros((self.n,self.d))
        U[0]=self.anchor_weight*Ja.T@Ja;u[0]=-self.anchor_weight*Ja.T@a
        return LinearGraph(self.n,self.d,self.i,self.j,Hii,Hij,Hjj,bi,bj,U,u,float(cost))


@dataclass
class LinearGraph:
    n:int;d:int;i:np.ndarray;j:np.ndarray
    Hii:np.ndarray;Hij:np.ndarray;Hjj:np.ndarray
    bi:np.ndarray;bj:np.ndarray;U:np.ndarray;u:np.ndarray;cost:float
    def __post_init__(self):
        self.D=self.U.copy();np.add.at(self.D,self.i,self.Hii);np.add.at(self.D,self.j,self.Hjj)
        self.b=self.u.copy();np.add.at(self.b,self.i,self.bi);np.add.at(self.b,self.j,self.bj)
        self.D=.5*(self.D+self.D.transpose(0,2,1))
    def sparse(self):
        d=self.d;rows=[];cols=[];data=[];nodes=np.arange(self.n)
        for a in range(d):
            for b in range(d):
                rows.extend([nodes*d+a,self.i*d+a,self.j*d+b])
                cols.extend([nodes*d+b,self.j*d+b,self.i*d+a])
                data.extend([self.D[:,a,b],self.Hij[:,a,b],self.Hij[:,a,b]])
        return sp.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(self.n*d,self.n*d)).tocsr()
    def directed(self):
        m=len(self.i)
        return tuple(np.ascontiguousarray(x) for x in (
            np.r_[self.i,self.j],np.r_[self.j,self.i],np.r_[np.arange(m,2*m),np.arange(m)],
            np.concatenate([self.Hii,self.Hjj]),np.concatenate([self.Hij,self.Hij.transpose(0,2,1)]),
            np.concatenate([self.Hjj,self.Hii]),np.concatenate([self.bi,self.bj]),np.concatenate([self.bj,self.bi])))


def odometry_initial(i,j,Rz,tz,n):
    k=tz.shape[1];R=np.tile(np.eye(k),(n,1,1));t=np.zeros((n,k));seen=np.zeros(n,bool);seen[0]=True
    # An odometry chain is used when present. Otherwise a measurement spanning tree.
    lookup={(int(a),int(b)):e for e,(a,b) in enumerate(zip(i,j))}
    if all((a,a+1) in lookup for a in range(n-1)):
        for a in range(n-1):
            e=lookup[(a,a+1)];R[a+1]=R[a]@Rz[e];t[a+1]=t[a]+R[a]@tz[e]
        return R,t
    for _ in range(n):
        changed=False
        for e,(a,b) in enumerate(zip(i,j)):
            if seen[a] and not seen[b]:
                R[b]=R[a]@Rz[e];t[b]=t[a]+R[a]@tz[e];seen[b]=True;changed=True
            elif seen[b] and not seen[a]:
                R[a]=R[b]@Rz[e].T;t[a]=t[b]-R[a]@tz[e];seen[a]=True;changed=True
        if seen.all():return R,t
        if not changed:raise ValueError('Disconnected graph')
    raise ValueError('Failed to initialize graph')


def load_compact_grid(root):
    arr=np.loadtxt(Path(root)/'smallGrid3D_edges.txt')
    if arr.ndim!=2 or arr.shape[1]!=9 or not np.isfinite(arr).all():raise ValueError('Invalid smallGrid compact fields')
    if not np.equal(arr[:,:2],np.floor(arr[:,:2])).all() or arr[:,:2].min()<0:raise ValueError('Invalid smallGrid endpoints')
    i=arr[:,0].astype(np.int64);j=arr[:,1].astype(np.int64)
    n=int(max(i.max(),j.max()))+1;tz=arr[:,2:5];Rz=Rotation.from_quat(arr[:,5:9]).as_matrix()
    O=np.tile(np.diag([100.,100.,100.,25.,25.,25.]),(len(arr),1,1))
    R,t=odometry_initial(i,j,Rz,tz,n)
    return PoseGraph('smallGrid3D',i,j,Rz,tz,O,R,t,source={'repo':'david-m-rosen/SE-Sync','path':'data/smallGrid3D.g2o','blob_sha':'2f3fb56d9fe43a43eba54c7a594f9729b585e3d1','initialization':'odometry chain','synthetic_standard_benchmark':True})


def load_g2o(path):
    """Full g2o parser for downloaded files; preserves every supplied information entry."""
    vs={};es=[];dim=None;fixed=[];edge_lines=[]
    for ln,line in enumerate(Path(path).read_text().splitlines(),1):
        z=line.split()
        if not z or z[0].startswith('#'):continue
        expected={'VERTEX_SE2':5,'VERTEX_SE3:QUAT':9,'EDGE_SE2':12,'EDGE_SE3:QUAT':31,'FIX':2}
        if z[0] not in expected or len(z)!=expected[z[0]]:raise ValueError(f'{path}:{ln}: unsupported tag or wrong field count')
        if z[0]=='FIX':fixed.append(int(z[1]));continue
        this_dim=2 if z[0].endswith('SE2') else 3
        if dim is not None and dim!=this_dim:raise ValueError(f'{path}:{ln}: mixed SE2/SE3 graph')
        dim=this_dim
        if not np.isfinite(np.array(z[1:],float)).all():raise ValueError(f'{path}:{ln}: nonfinite value')
        if z[0].startswith('VERTEX') and int(z[1]) in vs:raise ValueError(f'{path}:{ln}: duplicate vertex')
        if z[0]=='VERTEX_SE2':dim=2;vs[int(z[1])]=np.array(z[2:],float)
        elif z[0]=='VERTEX_SE3:QUAT':dim=3;vs[int(z[1])]=np.array(z[2:],float)
        elif z[0] in ('EDGE_SE2','EDGE_SE3:QUAT'):
            es.append((int(z[1]),int(z[2]),np.array(z[3:],float)));edge_lines.append(ln)
        else:raise ValueError(f'Unsupported g2o tag {z[0]}')
    if not vs or not es:raise ValueError(f'{path}: empty pose graph')
    if any(a not in vs or b not in vs for a,b,x in es):raise ValueError(f'{path}: edge references absent vertex')
    ids=sorted(vs);lookup={v:i for i,v in enumerate(ids)};i=np.array([lookup[a] for a,b,x in es]);j=np.array([lookup[b] for a,b,x in es]);n=len(ids)
    ed=np.array([x for a,b,x in es]);d=3 if dim==2 else 6;v=np.array([vs[z] for z in ids])
    if dim==2:Rz=rot2(ed[:,2]);tz=ed[:,:2];R=rot2(v[:,2]);t=v[:,:2];off=3
    else:Rz=Rotation.from_quat(ed[:,3:7]).as_matrix();tz=ed[:,:3];R=Rotation.from_quat(v[:,3:7]).as_matrix();t=v[:,:3];off=7
    O=np.zeros((len(es),d,d));a,b=np.triu_indices(d);O[:,a,b]=ed[:,off:];O[:,b,a]=ed[:,off:]
    return PoseGraph(Path(path).stem,i,j,Rz,tz,O,R,t,source={'file':str(path),'initialization':'supplied vertices','vertex_ids':ids,'edge_lines':edge_lines,'fixed_directives':fixed})


def normalize_linear(L):
    """Local change of coordinates x_i = S_i y_i; no objective damping.
    All diagonal node blocks become identity; all cross blocks are preserved.
    """
    ev,V=np.linalg.eigh(L.D)
    if ev.min()<=0:raise ArithmeticError('Cannot whiten a nonpositive node block')
    S=(V*(ev**-.5)[:,None,:])@V.transpose(0,2,1)
    Si=S[L.i];Sj=S[L.j]
    Q=LinearGraph(L.n,L.d,L.i,L.j,Si@L.Hii@Si,Si@L.Hij@Sj,Sj@L.Hjj@Sj,
                  np.einsum('eab,eb->ea',Si,L.bi),np.einsum('eab,eb->ea',Sj,L.bj),
                  S@L.U@S,np.einsum('nab,nb->na',S,L.u),L.cost)
    return Q,S


def eliminate_fixed_root(L):
    """Condition on delta_0=0 exactly; no large finite weight approximation.
    Original factors incident on the fixed root become unary factors. No new
    measurements or extra anchors are introduced.
    """
    mask=(L.i!=0)&(L.j!=0);U=L.U[1:].copy();u=L.u[1:].copy()
    for e in np.flatnonzero(L.i==0):U[L.j[e]-1]+=L.Hjj[e];u[L.j[e]-1]+=L.bj[e]
    for e in np.flatnonzero(L.j==0):U[L.i[e]-1]+=L.Hii[e];u[L.i[e]-1]+=L.bi[e]
    return LinearGraph(L.n-1,L.d,L.i[mask]-1,L.j[mask]-1,L.Hii[mask],L.Hij[mask],L.Hjj[mask],L.bi[mask],L.bj[mask],U,u,L.cost)
