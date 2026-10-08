# Benchmark inputs

| Name | Vertices | Original factors | File |
|---|---:|---:|---|
| smallGrid3D | 125 | 297 | `smallGrid3D_edges.txt` |
| FR079 | 989 | 1,217 | `FR079_P_toro.graph` |
| Sphere | 2,500 | 4,949 | `sphere2500.g2o` |
| Cubicle | 5,750 | 16,869 | `cubicle.g2o` |

FR079, Sphere, and Cubicle were copied from the public mirror
[nano-pgo, pinned commit b85562f](https://github.com/gisbi-kim/nano-pgo/tree/b85562f0acda4d3e556f5b2d27c14156336df75b/data).
The research audit compared these bytes with earlier sources:
[Luca Carlone's dataset page](https://lucacarlone.mit.edu/datasets/),
[OpenSLAM Vertigo Sphere](https://github.com/OpenSLAM-org/openslam_vertigo/blob/e57e88cbc1756070f37c6c32f87be947e7a72bb8/datasets/sphere2500/originalDataset/sphere2500.g2o), and
[SE-Sync data at commit 9b631b6](https://github.com/david-m-rosen/SE-Sync/tree/9b631b6b82d8aa7d32ab846412ae1c070412b7b6/data).

smallGrid3D is the public synthetic benchmark from `david-m-rosen/SE-Sync`, original path `data/smallGrid3D.g2o`, original blob SHA `2f3fb56d9fe43a43eba54c7a594f9729b585e3d1`. The compact text retains all 297 edge rows as `i j tx ty tz qx qy qz qw`. Constant information is represented implicitly by diagonal `[100,100,100,25,25,25]`. The compact file is a numerical transcription of original edges, not a byte-identical original g2o file, and its own SHA256 is recorded. Initialization is recomputed from measurements.

These source acknowledgments do not assert that raw file information matches the Lie-log residual convention used by this experiment. See [algorithm.md](../docs/algorithm.md#benchmark-model-and-transport) for the precise metric and Cubicle repair. The public dataset pages did not provide a separate named dataset license in the original audit. Preserve upstream academic attribution; this private archive supplies no new dataset license.

`frozen/` stores prepared **k0** solver arrays and their original audit metadata. They support fast trajectory replay; nonlinear commands rebuild the model from the input text. Verify packaged asset hashes with `python scripts/verify_manifest.py` from the repository root.
