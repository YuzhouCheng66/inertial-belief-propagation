"""Verify lossless packaged benchmark and result assets."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
assets=json.loads((ROOT/'docs/asset_manifest.json').read_text())
for asset in assets:
    path=ROOT/asset['path']
    assert path.stat().st_size==asset['bytes'],asset['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==asset['sha256'],asset['path']
print(f'Verified {len(assets)} packaged assets')
