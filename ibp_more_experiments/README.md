# Inertial Belief Propagation (I-BP)

Minimal research code for the **periodic Inertial Belief Propagation** experiments in three message domains:

1. Gaussian BP on a diffusion/PDE linear system,
2. non-Gaussian Ising BP,
3. complex tensor-network BP with positive-definite matrix messages.

The core schedule is

```text
8 accepted synchronous BP sweeps
        ↓
1 original-BP residual evaluation
        ↓
1 group-local residual + temporal correction
        ↓
write the correction back to the original messages
```

No global coarse solve or global direction combination is used.

## Main result

![BP vs I-BP](figures/bp_vs_ibp_2x2.png)

The three trajectory panels show the median over seeds 201/202/203. The horizontal axis counts **full BP-map evaluations**, including I-BP's extra residual proposal. The fourth panel compares wall-clock speedup when BP and I-BP use the same number of OpenMP threads.

| Problem | BP full updates | I-BP full updates | Reduction | 1-thread speedup | 4-thread speedup |
|---|---:|---:|---:|---:|---:|
| Gaussian/PDE | 13,009 | 1,800 | 6.84× | 5.78× | 6.41× |
| Non-Gaussian Ising | 12,193 | 1,215 | 10.03× | 9.89× | 9.68× |
| Complex tensor network | 6,169 | 1,107 | 5.57× | 4.86× | 4.84× |

These are deliberately slow BP instances. They demonstrate acceleration of the BP iteration itself; they are **not** claims of application-level state of the art against AMG/PCG, exact/region-based inference, CTMRG, boundary-MPS, etc.

Counts in this table are independently computed medians over seeds 201/202/203.
Reduction and speedup columns are medians of the **per-seed ratios**, which need
not equal the ratio of the displayed median counts or median times. The
trajectory CSV contains aggregated median curves; its noninteger evaluation
coordinates are not individual solver checkpoints. The binary fixtures and
historical measurements are preserved as supplied.

This directory is the additional-experiment archive within the main
[Inertial Belief Propagation repository](../README.md). Its beta cap is 0.95;
the SE2/SE3 solver in the parent uses 0.995 and a different local correction.

## Repository layout

```text
.
├── src/
│   └── ibp_parallel.cpp       # C++17/OpenMP BP + I-BP implementation
├── data/
│   ├── check_*.bin            # tiny smoke-test fixtures
│   └── *s201/202/203*.bin     # the 9 main benchmark fixtures
├── scripts/
│   ├── reproduce.py           # smoke test / full timing runs
│   └── plot_results.py        # recreates the 2×2 summary figure
├── results/
│   ├── per_case.csv           # per-seed measured results
│   ├── summary.csv            # compact three-row summary
│   └── trajectories.csv       # median trajectories used by the figure
├── figures/
│   └── bp_vs_ibp_2x2.png
├── Makefile
└── requirements.txt
```

The binary fixtures are included so the experiments run without a separate data-generation pipeline.

## Build

Linux with a C++17 compiler and OpenMP:

```bash
make
```

Equivalent command:

```bash
g++ -O3 -std=c++17 -fopenmp src/ibp_parallel.cpp -o build/ibp_parallel
```

Python plotting dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Verified smoke test

```bash
make smoke
```

This runs **BP and I-BP** on small Gaussian, Ising, and tensor-network fixtures and requires all six runs to reach their native stopping criteria.

Expected final line:

```text
SMOKE TEST PASSED: Gaussian, Ising, and tensor-network BP/I-BP all converged.
```

## Reproduce the benchmark timings

```bash
make reproduce
```

This runs the nine main fixtures with BP and I-BP using 1, 2, and 4 OpenMP threads, with three timing repeats per configuration. Runtime depends on the CPU. The published CSVs in `results/` are the measurements used for the included figure.

For a shorter run:

```bash
python3 scripts/reproduce.py --threads 1 4 --repeats 1
```

## Recreate the figure

```bash
make plot
```

This regenerates:

```text
figures/bp_vs_ibp_2x2.png
```

from the checked-in CSV results.

## Implementation details

### Gaussian/PDE

The precision messages are pre-converged and frozen in the fixture. The online state is the information vector. After eight sweeps, group `g` forms a local physical residual and applies

```text
delta_g = 0.45 * diag(A_gg)^(-1) * residual_g
          + beta_g * temporal_direction_g
```

and lifts the state correction back into the original information-vector messages using the incoming message precisions.

### Ising

The executable uses the exact nonlinear binary Ising cavity update. The inertial correction acts directly on the scalar cavity-field messages. No Gaussian approximation is used.

### Tensor network

Each directed message is a 2×2 Hermitian positive-definite matrix represented by its three Pauli/Bloch coordinates. The correction is performed in a trace-free matrix-log coordinate and mapped back to a normalized positive-definite message.

### Parallelism

A single graph is distributed across OpenMP threads. All reported comparisons use the same synchronous algorithmic trajectory for a given method; changing the number of threads changes execution time, not the mathematical update sequence.

## Reproducibility notes

- FP64 throughout.
- Synchronous BP updates.
- I-BP period: 8 accepted BP sweeps.
- Local residual gain: `tau = 0.45`.
- Inertial cap: `beta_max = 0.95`.
- Main seeds: 201, 202, 203.
- Gaussian stopping tolerance: `1e-8` for both true linear residual and BP message residual.
- Ising/tensor stopping tolerance: `1e-10` for the original BP fixed-point residual.
- Timings are hardware dependent; update counts are the more portable algorithmic metric.

## Status

Research prototype. A public license has intentionally **not** been selected in this package; choose one before publishing the repository if desired.
