# Provenance and packaging

The core method is the original `projected995` experiment: 8 mean sweeps, balanced block residual correction, local projected inertia, independent adaptive group beta with cap .995, no filter. The exported module functions derive from these research files:

| Repository module | Research source |
|---|---|
| `lie.py` | `work/study/code/lie_graph.py` |
| `models.py` | `work/study/code/new_models.py` |
| `gbp.py` | selected frozen GBP kernels in `gbp_core.py` |
| `diagnostics.py`, `precision.py` | `message_diagnostics.py`, `accurate_prepare.py` |
| `precision.py`, `warp.py` | fused precision kernel and affine warp in `work/outer20_cycle_20261004/engine.py` |
| `preconditioner.py` | `phase3_preconditioner.py`, `_step` from `phase4_inertial_gbp.py` |
| `solver.py` | unfiltered mode-1/mode-0 path of `side_research/edge_inertia_20261003/experiment.py` |
| `nonlinear.py` | original 20-complete-cycle outer loop in `work/outer20_cycle_20261004/run.py` |

One-time source selection produced normal Python modules with relative imports. There is no runtime AST extraction, hidden import from another workspace, user-specific absolute path, or lookup of unpublished research code. The portable wrapper fixes group size 32, period 8, beta cap .995, spatial scale 1, and damping 1 for mean sweeps. Historical `projected995`/`GBP` result labels are preserved; the CLI calls them `ibp`/`gbp`.

The frozen caches contain a lossless subset of the original arrays required to replay the mean solver: graph blocks, directed frozen operators, eta lift, coordinate maps, and local correction factors. Duplicated factor/preparation arrays are omitted. Each source NPZ SHA256 and original preparation metadata are preserved; `scripts/verify_manifest.py` verifies every packaged source/data/result asset against `docs/asset_manifest.json`.

Original source and experiment SHA256 values are in [source_manifest.json](source_manifest.json). Packaged asset hashes are in [asset_manifest.json](asset_manifest.json). Input dataset origins and the smallGrid compact representation are documented in [data/README.md](../data/README.md). The original audit metadata and traces are kept separate from packaging checks and fresh output directories.

The English-label figures were regenerated with `scripts/plot_results.py` from the archived comparison arrays and costs. They retain actual recorded points and do not introduce smoothing, inferred timing, or extrapolated convergence.

Precision JIT functions retain the original nonlinear engine's `cache=False` policy. Enabling disk caching during packaging caused a native access violation on a later process restart with Windows/Numba 0.59.1; the cache was disabled and repeated-process checks rerun. This affects compilation/startup overhead, not the precision equations or solver update.

Packaging does not add a global synchronization coefficient, a new grouping scheme, pseudo edges, message reconstruction, or a full-cycle energy filter. Later exploratory variants are outside this repository's core method and result comparisons.

## Additional experiments

The user-supplied `ibp_github_minimal` package was copied into the repository as
`ibp_more_experiments/`, without an extra wrapper directory. Its C++ kernel,
binary inputs, original CSV measurements, and requested `bp_vs_ibp_2x2.png`
figure are preserved. The subdirectory README adds parent-repository context
and clarifies median-ratio statistics; the original source hashes are included
under `provided_more_experiments/` in the source manifest. Added audit scripts
verify fixture structure, topology, statistical summaries, work accounting,
and native thread determinism. The Linux CI job builds the C++/OpenMP executable
and runs its six original convergence smoke tests.

These three domains use beta cap .95 and gain .45. They share the periodic,
group-local inertial idea but do not reuse the SE2/SE3 physical low-mode
preconditioner. Their plotting CSV stores aggregated median curves with
noninteger coordinates; the SE2/SE3 figure separately uses original discrete
eight-sweep checkpoints.
