"""Audit additional binary fixtures, CSV ratios, and fixture provenance."""
from pathlib import Path
import csv
import json
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'ibp_more_experiments'
checks=[]
for path in sorted((ROOT/'data').glob('*.bin')):
    raw=path.read_bytes()
    header=np.frombuffer(raw,dtype='<i4',count=7)
    magic,domain,L,N,G,seed,mode=map(int,header)
    assert magic==734211 and domain in (0,1,2) and N==L*L and G>0
    position=36
    def take(dtype,count):
        global position
        out=np.frombuffer(raw,dtype=dtype,count=count,offset=position)
        position+=np.dtype(dtype).itemsize*count
        return out
    neighbors=take('<i4',4*N).reshape(N,4)
    offsets=take('<i4',G+1);order=take('<i4',N)
    assert offsets[0]==0 and offsets[-1]==N and np.all(np.diff(offsets)>0)
    assert np.array_equal(np.sort(order),np.arange(N))
    assert np.all((neighbors>=-1)&(neighbors<N))
    for slot in range(4):
        nodes=np.flatnonzero(neighbors[:,slot]>=0)
        assert np.array_equal(neighbors[neighbors[nodes,slot],(slot+2)%4],nodes)
    field=take('<f8',N);prior=take('<f8',N);diag=take('<f8',N)
    p=take('<f8',4*N);P=take('<f8',N);alpha=take('<f8',4*N)
    if domain==2:tensor=take('<f8',N*256);assert np.isfinite(tensor).all()
    assert position==len(raw)
    assert all(np.isfinite(x).all() for x in (field,prior,diag,p,P,alpha))
    if domain==0:
        assert np.all(P>0) and np.all(p>=0)
        np.testing.assert_allclose(P,prior+p.reshape(N,4).sum(axis=1),rtol=1e-12,atol=1e-12)
    metadata=path.with_suffix('.json')
    if metadata.exists():
        data=json.loads(metadata.read_text())
        assert data['L']==L and data['N']==N and data['seed']==seed
        assert data['beta_cap']==.95 and data['tau']==.45 and data['period']==8
    checks.append(dict(fixture=path.name,domain=domain,nodes=N,groups=G,seed=seed,bytes=len(raw)))
with (ROOT/'results/per_case.csv').open(newline='') as f:per=list(csv.DictReader(f))
with (ROOT/'results/summary.csv').open(newline='') as f:summary=list(csv.DictReader(f))
assert len(per)==9 and len(summary)==3
for domain,row in enumerate(summary):
    cases=[r for r in per if int(r['domain'])==domain]
    assert len(cases)==3
    def median(key):return statistics.median(float(r[key]) for r in cases)
    for label,key in (('bp_sweeps','BP_sweeps'),('ibp_sweeps','IBP_sweeps'),
        ('ibp_corrections','IBP_corrections'),('bp_full_evaluations','BP_evals'),
        ('ibp_full_evaluations','IBP_evals'),('full_evaluation_reduction_x','call_reduction'),
        ('speedup_1thread_x','algorithm_single'),('speedup_4thread_x','algorithm_four')):
        assert abs(float(row[label])-median(key))<1e-10,(label,row['problem'])
    for case in cases:
        assert int(case['BP_evals'])==int(case['BP_sweeps'])+1
        assert int(case['IBP_evals'])==9*(int(case['IBP_sweeps'])//8)
        assert int(case['IBP_corrections'])==int(case['IBP_sweeps'])//8-1
        assert abs(float(case['call_reduction'])-float(case['BP_evals'])/float(case['IBP_evals']))<1e-10
        for threads in (1,4):
            assert float(case[f'BP_{threads}'])>0 and float(case[f'IBP_{threads}'])>0
with (ROOT/'results/trajectories.csv').open(newline='') as f:trajectories=list(csv.DictReader(f))
assert {r['domain'] for r in trajectories}=={r['problem'] for r in summary}
assert {r['method'] for r in trajectories}=={'BP','I-BP'}
for row in trajectories:
    assert abs(float(row['residual_over_tolerance'])-float(row['residual'])/float(row['tolerance']))<1e-7*max(1.,float(row['residual_over_tolerance']))
print(json.dumps(dict(fixtures_pass=True,summary_ratio_medians_pass=True,work_ledger_pass=True,
    trajectory_scaling_pass=True,fixtures=checks),indent=2))
