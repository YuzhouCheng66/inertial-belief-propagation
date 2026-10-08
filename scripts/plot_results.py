"""Regenerate scientific comparison figures from actual saved points."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT/'runs/figures')
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
with np.load(ROOT/'results/frozen/comparison.npz') as z:
    for column,title,filename in ((2,'Normal relative residual','frozen_residuals.png'),(5,'Relative quadratic energy gap','frozen_energy.png')):
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for ax,name in zip(axes.ravel(),('smallGrid3D','FR079','Sphere','Cubicle')):
            for key,label in (('pure','GBP'),('unfiltered','Local IBP')):
                points=z[f'{name}_{key}'];ax.loglog(points[:,0],np.maximum(points[:,column],1e-18),label=label)
            ax.set(title=name,xlabel='Actual GBP mean sweeps',ylabel=title);ax.grid(alpha=.25);ax.legend()
        fig.tight_layout();fig.savefig(args.output/filename,dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,name in zip(axes,('FR079','Cubicle')):
    for method,label in (('GBP','GBP'),('projected995','Local IBP')):
        data=json.loads((ROOT/f'results/nonlinear/{name}_{method}/results.json').read_text())
        costs=[data['initial_cost']]+[r['cost_after'] for r in data['rows'] if 'cost_after' in r]
        ax.semilogy(np.arange(len(costs)),costs,marker='.',label=label)
    ax.set(title=name,xlabel='Accepted relinearizations (20 complete cycles each)',ylabel='Nonlinear cost');ax.grid(alpha=.25);ax.legend()
fig.tight_layout();fig.savefig(args.output/'nonlinear_cost.png',dpi=180);plt.close(fig)
print(args.output)
