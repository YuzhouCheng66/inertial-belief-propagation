"""Explicit benchmark model: right Lie-log residual, Huber 5, fixed root."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from numba import njit
from .lie import PoseGraph, LinearGraph, residual_jacobian, load_compact_grid, load_g2o, normalize_linear, eliminate_fixed_root
ROOT=Path(__file__).resolve().parents[2]

def skew(v):
 v=np.asarray(v);a=np.zeros(v.shape[:-1]+(3,3));a[...,0,1]=-v[...,2];a[...,0,2]=v[...,1];a[...,1,0]=v[...,2];a[...,1,2]=-v[...,0];a[...,2,0]=-v[...,1];a[...,2,1]=v[...,0];return a


def rot2(a):
 a=np.asarray(a);R=np.empty(a.shape+(2,2));R[...,0,0]=R[...,1,1]=np.cos(a);R[...,0,1]=-np.sin(a);R[...,1,0]=np.sin(a);return R


def make_odometry(i,j,Rz,tz,n,policy='sequential',return_metadata=False):
 """Compose a deterministic measurement tree, with an explicit topology policy.

 ``sequential`` requires the consecutive-ID chain. ``sequential_preferred``
 keeps that chain when connected and otherwise joins its components with the
 shortest ID-gap measurements (edge order breaks ties). This is initialization,
 not an optimization step; every nonsequential tree edge is reported.
 """
 if policy not in ('sequential','sequential_preferred'):
  raise ValueError(f'Unknown initialization policy: {policy}')
 k=tz.shape[1];R=np.repeat(np.eye(k)[None],n,axis=0);t=np.zeros((n,k));known=np.zeros(n,bool);known[0]=True
 parent=np.arange(n)
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 selected=[]
 order=np.flatnonzero(np.abs(i-j)==1).tolist()
 if policy=='sequential_preferred':
  order+=sorted(np.flatnonzero(np.abs(i-j)!=1).tolist(),key=lambda e:(abs(int(i[e])-int(j[e])),e))
 for e in order:
  a,b=find(int(i[e])),find(int(j[e]))
  if a!=b:parent[b]=a;selected.append(e)
 if len(selected)!=n-1:
  label='Sequential odometry' if policy=='sequential' else 'Measurement graph'
  raise ValueError(f'{label} does not connect all poses ({n-len(selected)} components)')
 adj=[[] for _ in range(n)]
 for e in selected:
  a,b=int(i[e]),int(j[e]);adj[a].append((b,e,False));adj[b].append((a,e,True))
 q=[0]
 for a in q:
  for b,e,rev in adj[a]:
   if known[b]:continue
   re=Rz[e].T if rev else Rz[e];te=-Rz[e].T@tz[e] if rev else tz[e]
   R[b]=R[a]@re;t[b]=t[a]+R[a]@te;known[b]=True;q.append(b)
 metadata=dict(policy=policy,tree_edges=selected,nonsequential_tree_edges=[dict(edge_index=e,i=int(i[e]),j=int(j[e])) for e in selected if abs(int(i[e])-int(j[e]))!=1],root=0)
 return (R,t,metadata) if return_metadata else (R,t)


@njit(cache=True)
def jr_inv(x):
 n,d=x.shape;out=np.empty((n,d,d));I=np.eye(d)
 for k in range(n):
  A=np.zeros((d,d))
  if d==3:
   A[0,1]=-x[k,2];A[1,0]=x[k,2];A[0,2]=x[k,1];A[1,2]=-x[k,0]
  else:
   for s in (0,3):
    A[s,s+1]=-x[k,5];A[s,s+2]=x[k,4];A[s+1,s]=x[k,5];A[s+1,s+2]=-x[k,3];A[s+2,s]=-x[k,4];A[s+2,s+1]=x[k,3]
   A[0,4]=-x[k,2];A[0,5]=x[k,1];A[1,3]=x[k,2];A[1,5]=-x[k,0];A[2,3]=-x[k,1];A[2,4]=x[k,0]
  term=I.copy();J=I.copy()
  for m in range(1,60):
   term=-(term@A)/(m+1.);J+=term
   if np.max(np.abs(term))<2e-16*max(1.,np.max(np.abs(J))):break
  out[k]=np.linalg.solve(J,I)
 return out


def adj(R,t):
 n,k=t.shape;d=3 if k==2 else 6;A=np.zeros((n,d,d));A[:,:k,:k]=R
 if k==2:A[:,0,2]=t[:,1];A[:,1,2]=-t[:,0];A[:,2,2]=1.
 else:A[:,:3,3:]=skew(t)@R;A[:,3:,3:]=R
 return A


def log_pose(R,t):
 n,k=t.shape
 if k==2:
  a=np.arctan2(R[:,1,0],R[:,0,0]);s=np.abs(a)<1e-4;c=np.empty(n)
  c[s]=1-a[s]**2/12-a[s]**4/720;c[~s]=.5*a[~s]/np.tan(.5*a[~s])
  return np.c_[c*t[:,0]+.5*a*t[:,1],c*t[:,1]-.5*a*t[:,0],a]
 p=Rotation.from_matrix(R).as_rotvec();th=np.linalg.norm(p,axis=1);tt=th*th;s=th<1e-4;c=np.empty(n)
 c[s]=1/12+tt[s]/720+tt[s]**2/30240;c[~s]=(1-.5*th[~s]/np.tan(.5*th[~s]))/tt[~s]
 K=skew(p);return np.c_[np.einsum('nij,nj->ni',np.eye(3)-.5*K+c[:,None,None]*(K@K),t),p]


def log_residual(Ri,ti,Rj,tj,Rz,tz,jac=True):
 RT=Ri.transpose(0,2,1);ZT=Rz.transpose(0,2,1);Rp=RT@Rj;tp=np.einsum('nij,nj->ni',RT,tj-ti)
 e=log_pose(ZT@Rp,np.einsum('nij,nj->ni',ZT,tp-tz))
 if not jac:return e,None,None
 Jj=jr_inv(e);Rinv=Rp.transpose(0,2,1);tinv=-np.einsum('nij,nj->ni',Rinv,tp)
 return e,-Jj@adj(Rinv,tinv),Jj


def load_toro(path):
 vs={};edges=[]
 for line in Path(path).read_text().splitlines():
  a=line.split()
  if not a:continue
  if a[0]=='VERTEX2':vs[int(a[1])]=list(map(float,a[2:5]))
  elif a[0]=='EDGE2':edges.append((int(a[1]),int(a[2]),list(map(float,a[3:]))))
 ids=sorted(vs);mp={v:i for i,v in enumerate(ids)};v=np.array([vs[z] for z in ids]);e=np.array([x[2] for x in edges]);i=np.array([mp[x[0]] for x in edges]);j=np.array([mp[x[1]] for x in edges]);O=np.zeros((len(e),3,3))
 O[:,0,0]=e[:,3];O[:,0,1]=O[:,1,0]=e[:,4];O[:,1,1]=e[:,5];O[:,2,2]=e[:,6];O[:,0,2]=O[:,2,0]=e[:,7];O[:,1,2]=O[:,2,1]=e[:,8]
 return PoseGraph('FR079',i,j,rot2(e[:,2]),e[:,:2],O,rot2(v[:,2]),v[:,:2])


def raw_load(name, data_dir=None):
 root=Path(data_dir) if data_dir is not None else ROOT/'data'
 if name=='smallGrid3D':return load_compact_grid(root)
 if name=='FR079':return load_toro(root/'FR079_P_toro.graph')
 return load_g2o(root/dict(Sphere='sphere2500.g2o',Cubicle='cubicle.g2o')[name])


class Graph:
 def __init__(self,name,mode='log_raw',huber=5.,floor=None,initialization='sequential_preferred',data_dir=None):
  if mode not in ('native','log_raw','toro_native'):raise ValueError(f'Unknown residual mode: {mode}')
  if mode=='toro_native' and name!='FR079':raise ValueError('toro_native is defined only for the FR079 TORO input')
  if not np.isfinite(huber) or huber<0:raise ValueError('Huber threshold must be finite and nonnegative')
  if floor is not None and (not np.isfinite(floor) or floor<=0):raise ValueError('Information eigenvalue floor must be finite and positive')
  g=raw_load(name,data_dir);self.name=name;self.n=g.n;self.d=g.d;self.i=g.i;self.j=g.j;self.Rz=g.Rz;self.tz=g.tz;self.mode=mode;self.huber=float(huber);self.floor=floor
  self.rawOmega=g.Omega.copy();self.Omega=g.Omega.copy()
  if not np.isfinite(self.Omega).all():raise ValueError(f'{name}: nonfinite input precision')
  metric_transform='none: original file information paired with selected residual'
  if mode=='toro_native':
   # TORO weights source-frame translation error; native g2o rotates that
   # error by Rz.T. Congruence preserves the original TORO objective exactly.
   B=np.zeros_like(self.Omega);B[:,:2,:2]=self.Rz;B[:,2,2]=1.
   self.Omega=B.transpose(0,2,1)@self.Omega@B
   metric_transform='Omega_g2o = B.T @ Omega_TORO @ B, B = diag(Rz, 1)'
  ev,V=np.linalg.eigh(self.Omega)
  if floor is not None:self.Omega=(V*np.maximum(ev,float(floor))[:,None,:])@V.transpose(0,2,1)
  elif ev.min()<=0:raise ValueError(f'{name}: invalid input precision; explicit repair needed')
  self.precision_metadata=dict(policy='unchanged' if floor is None else 'eigenvalue_floor',floor=floor,changed_edges=0 if floor is None else int(np.any(ev<floor,axis=1).sum()),raw_minimum=float(ev.min()),coordinate_transform=metric_transform)
  if initialization=='supplied':
   # Change the global reference frame only, so the fixed root is identity.
   RT=g.R0[0].T;self.R0=RT@g.R0;self.t0=np.einsum('ab,nb->na',RT,g.t0-g.t0[0]);self.initialization_metadata=dict(policy='supplied',root=0)
  else:self.R0,self.t0,self.initialization_metadata=make_odometry(self.i,self.j,self.Rz,self.tz,self.n,initialization,True)
  self.model_metadata=dict(residual=mode,coordinates='right SE exponential, translation then rotation',huber=self.huber,root='exactly fixed pose 0',initialization=self.initialization_metadata,precision=self.precision_metadata)
 def evaluate(self,R,t,jac=False):
  args=(R[self.i],t[self.i],R[self.j],t[self.j],self.Rz,self.tz)
  e,Ji,Jj=log_residual(*args,jac) if self.mode=='log_raw' else residual_jacobian(*args)
  s2=np.einsum('ea,eab,eb->e',e,self.Omega,e);s=np.sqrt(np.maximum(s2,0))
  if self.huber>0:
   w=np.minimum(1.,self.huber/np.maximum(s,1e-300));cost=np.where(s<=self.huber,.5*s2,self.huber*s-.5*self.huber**2).sum()
  else:w=np.ones_like(s);cost=.5*s2.sum()
  if jac:return float(cost),e,Ji,Jj,w,s
  return float(cost)
 def assemble(self,R,t):
  cost,e,Ji,Jj,w,s=self.evaluate(R,t,True);C=np.linalg.cholesky(self.Omega*w[:,None,None]).transpose(0,2,1)
  Fi=C@Ji;Fj=C@Jj;we=np.einsum('eab,eb->ea',C,e)
  Hii=Fi.transpose(0,2,1)@Fi;Hij=Fi.transpose(0,2,1)@Fj;Hjj=Fj.transpose(0,2,1)@Fj
  bi=-np.einsum('eab,ea->eb',Fi,we);bj=-np.einsum('eab,ea->eb',Fj,we)
  raw=eliminate_fixed_root(LinearGraph(self.n,self.d,self.i,self.j,Hii,Hij,Hjj,bi,bj,np.zeros((self.n,self.d,self.d)),np.zeros((self.n,self.d)),cost))
  L,S=normalize_linear(raw);mask=(self.i!=0)&(self.j!=0);Fi=Fi[mask]@S[L.i];Fj=Fj[mask]@S[L.j];offset=L.directed()[-1]
  shifted=LinearGraph(L.n,L.d,L.i,L.j,L.Hii,L.Hij,L.Hjj,np.zeros_like(L.bi),np.zeros_like(L.bj),L.U,L.b,cost)
  return shifted,S,np.ascontiguousarray(Fi),np.ascontiguousarray(Fj),offset,raw,dict(downweighted=int((w<1).sum()),min_weight=float(w.min()),raw_cost=float(.5*np.sum(s*s)))
