# Increasing-Concave-Mappings-part-II
# ICM-Based Newton Method for Large-Scale AC Power Flow

This repository provides MATLAB codes for validating the proposed Increasing-Concave Mapping (ICM)-based Newton method for AC power flow analysis. The simulations investigate convergence properties, voltage iteration trajectories, and computational efficiency in large-scale AC networks.

## Repository Contents

- **ICMpart2_three_cases_123.m**: Tests three 123-bus radial AC microgrid cases:
  - Case 1: Lossless network with PV buses.
  - Case 2: Homogeneous R/X network with PV buses.
  - Case 3: Nonuniform R/X network without PV buses.

  Compares ICM with conventional Polar Newton and Complex Current Newton methods under different initializations, including flat start and DC start.

- **run_ac10000.m**: Large-scale validation using a 10,000-bus AC benchmark. Compares:
  - Continuous Power Flow (CPF).
  - ICM-based bisection search.
  - Eight-worker parallel ICM nine-section search.

- **ieee10000.mat**: 10,000-bus benchmark data in MATPOWER format.

## Requirements

- MATLAB
- Optimization Toolbox (for `fsolve` is not required; the implementation uses sparse linear solvers).
- Parallel Computing Toolbox (for eight-worker parallel execution).

## How to Run

Place all files in the same folder and run:

```matlab
ICMpart2_three_cases_123
```

For the 10,000-bus benchmark, run:

```matlab
results = run_ac10000();
```


## Generated Outputs

**123-bus cases**
- Convergence comparison of different Newton methods.
- Power mismatch trajectories.
- Voltage-magnitude iteration trajectories for all PQ buses across the three cases.
- Iteration counts, convergence status, final power-flow residuals, and voltage ranges.

**10,000-bus benchmark**
- Maximum loading factor obtained by CPF, ICM bisection, and parallel ICM nine-section search.
- Repeated computational times and median execution times.
- Power-flow residuals and loading-factor accuracy checks.
- Numerical verification of the critical loading point.

The scripts display numerical results and figures directly in MATLAB.
