"""Common nonlinear outer loop: 20 complete cycles and affine message warping."""
import json
import time
from pathlib import Path
import numpy as np
from .lie import retract
from .models import Graph
from .precision import prepare
from .preconditioner import build_local_pc
from .solver import solve, empty_pc
from .warp import warp


def load_case(name,data_dir=None):
    return Graph(name,'log_raw',5.,1. if name=='Cubicle' else None,data_dir=data_dir)


def run(name,method,output,*,outer=20,data_dir=None):
    """Reproduce the archived outer loop; output must be a new/empty directory."""
    if method not in ('ibp','gbp'):raise ValueError('method must be ibp or gbp')
    if not isinstance(outer,int) or outer<1:raise ValueError('outer must be positive')
    target=Path(output)
    if target.exists() and any(target.iterdir()):raise ValueError('output directory must be empty')
    target.mkdir(parents=True,exist_ok=True)
    g=load_case(name,data_dir);R=g.R0.copy();t=g.t0.copy();previous=None
    rows=[];Rs=[R.copy()];ts=[t.copy()];traces=[];beta_histories=[]
    status='outer_cap';start=time.perf_counter()
    result=dict(name=name,method=method,outer_cap=outer,cycles_per_linearization=20,
        sweeps_per_cycle=8,bcap=.995,filter_rounds=0,group=32,precision_tol=1e-10,
        precision_damping=.5,model=g.model_metadata,initial_cost=float(g.evaluate(R,t)),
        eta_initialization='canonical zero at k0; affine precision/canonical eta pullback later',
        inertia_policy='independent beta and history reset at each relinearization',
        cycle_end='20 complete cycles, including correction 20',
        reductions='local inner directions; global norms for audits and global outer Armijo cost/b_dot_step',rows=rows)
    def save():
        result.update(status=status,accepted_steps=len(Rs)-1,final_cost=float(g.evaluate(R,t)),
            elapsed_seconds=time.perf_counter()-start,
            totals={key:sum(r.get(key,0) for r in rows) for key in
                ['eta_sweeps','precision_sweeps','F_calls','corrections','linearize_seconds',
                 'precision_seconds','basis_seconds','warp_seconds','inner_seconds','linesearch_seconds']})
        (target/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        np.savez_compressed(target/'poses.npz',R=np.array(Rs),t=np.array(ts))
        if traces:np.savez_compressed(target/'inner_traces.npz',trace=np.array(traces),beta_history=np.array(beta_histories))
    for it in range(outer):
        row=dict(outer=it+1)
        stamp=time.perf_counter();L,S,Fi,Fj,offset,raw,_=g.assemble(R,t)
        row.update(linearize_seconds=time.perf_counter()-stamp,cost_before=float(L.cost),
            gradient_measure=float(np.linalg.norm(L.b)/max(1.,np.sqrt(2*L.cost))))
        if row['gradient_measure']<=1e-6:status='stationary';break
        p0=None;eta0=-offset.copy();wm={}
        if previous is not None:p0,eta0,wm=warp(previous,S,offset,L.directed()[1],R,t)
        row.update(warp=wm,warp_seconds=wm.get('seconds',0.))
        print(f'Preparing {name} {method} linearization {it+1}, cost={L.cost:.12g}',flush=True)
        P=prepare(L,Fi,Fj,p0=p0,eta0=eta0,joint=False,tol=1e-10,cap=20000,initial='upper',omega=.5)
        row.update(precision_sweeps=P['precision_sweeps'],precision_seconds=P['prepare_seconds'],
            precision_change=P['precision_change'],lift_identity_error=P['lift_identity_error'],
            incoming_rank_deficient=P['incoming_rank_deficient'])
        if method=='ibp':
            pc,_,pcm=build_local_pc(L,raw,S,group=32);row['basis_seconds']=pcm['setup_seconds']
        else:pc=empty_pc(L);row['basis_seconds']=0.
        out=solve(L,P,pc,offset,eta=P['eta0'],method=method,sweeps=160,tol=0.,complete_cycles=True)
        eta,x,trace,im=out['eta'],out['x'],out['trace'],out['metrics']
        traces.append(trace);beta_histories.append(out['beta_history'])
        row.update(eta_sweeps=im['sweeps'],F_calls=im['F_calls'],corrections=im['corrections'],
            inner_seconds=im['seconds'],normal_rr=im['normal_residual'],message_rr=im['message_residual'],
            restarts=im['restarts'],mean_beta=im['mean_beta'],max_beta=im['max_beta'])
        if im['status']=='diverged':rows.append(row);status='inner_diverged';break
        bd=float(np.sum(L.b*x));row['b_dot_step']=bd
        if not np.isfinite(bd) or bd<=0:rows.append(row);status='not_descent';break
        dx=np.zeros((g.n,g.d));dx[1:]=np.einsum('nij,nj->ni',S,x)
        accepted=False;stamp=time.perf_counter()
        for bt in range(30):
            alpha=2.**(-bt);rn,tn=retract(R,t,dx,alpha);cost=g.evaluate(rn,tn)
            if np.isfinite(cost) and cost<=L.cost-1e-4*alpha*bd:accepted=True;break
        row.update(linesearch_seconds=time.perf_counter()-stamp,backtracks=bt)
        if not accepted:rows.append(row);status='line_search_failed';break
        row.update(cost_after=float(cost),alpha=alpha,physical_step_norm=float(np.linalg.norm(alpha*dx)))
        previous=(P['p'].copy(),eta.copy(),offset.copy(),S.copy(),R.copy(),t.copy())
        R,t=rn,tn;Rs.append(R.copy());ts.append(t.copy());rows.append(row)
        np.savez_compressed(target/'state.npz',p=P['p'],eta=eta,offset=offset,S=S,
            oldR=previous[4],oldt=previous[5],R=R,t=t,x=x)
        save()
        print(f'Accepted {it+1}: cost={cost:.12g}, alpha={alpha:g}, mean_sweeps=160, corrections={im["corrections"]}',flush=True)
    save()
    return result
