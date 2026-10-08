"""Full original capped comparisons; fresh outputs stay outside archived results."""
import argparse
import subprocess
import sys
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--datasets',nargs='+',default=['smallGrid3D','FR079','Sphere','Cubicle'])
args=parser.parse_args()
caps={'smallGrid3D':(200000,80000),'FR079':(800000,80000),
      'Sphere':(400000,80000),'Cubicle':(400000,80000)}
for name in args.datasets:
    for method,cap in zip(('gbp','ibp'),caps[name]):
        subprocess.run([sys.executable,'-m','inertial_bp','frozen','--dataset',name,
            '--method',method,'--sweeps',str(cap),'--tol','1e-6',
            '--output',str(args.output/f'{name}_{method}')],check=True)
