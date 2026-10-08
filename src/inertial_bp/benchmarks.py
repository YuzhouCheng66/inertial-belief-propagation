"""Load the compact, lossless subset of the original frozen k0 caches."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def load_frozen(name,cache_dir=None):
    if name not in ('smallGrid3D','FR079','Sphere','Cubicle'):raise ValueError('unknown dataset')
    root=Path(cache_dir) if cache_dir is not None else ROOT/'data/frozen'
    with np.load(root/f'{name}_k0.npz',allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    L=SimpleNamespace(n=int(a['n']),d=int(a['d']),**{k:a[k] for k in ('i','j','D','Hij','b','u')})
    P={k[2:]:v for k,v in a.items() if k.startswith('P_')}
    pc=tuple(a[f'pc_{k}'] for k in range(5))
    return L,P,pc,a['offset']
