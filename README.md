# Variational Monte Carlo (VMC) Toy Systems

A compact, single-file Variational Monte Carlo (VMC) in Python.  
It demonstrates the full workflow you’d use in computational physics research code:

- **Metropolis–Hastings MCMC** sampling from \|ψ(x;θ)\|² using a **log-density** for numerical stability
- **Finite-difference Laplacian** (2nd / 4th order) to build the kinetic term and local energy
- **Correlation-aware uncertainty** via integrated autocorrelation time **τ_int** and **effective sample size (ESS)**
- **Stochastic gradient optimisation** of variational parameters using the standard VMC gradient estimator

---

## What’s included

### Toy systems (in `vmc.py`)
- **1D harmonic oscillator (HO)**: sanity checks on known eigen-energies
- **3D hydrogen-like atom (1s trial)**: optimise a single variational parameter θ
- **6D two-electron diatomic toy model**: one-shot sampling + symmetry-aware projected density plot

### Diagnostics
Printed per run (where applicable):
- acceptance rate
- τ_int
- ESS
- energy estimate ± correlation-aware standard error

---

## Quickstart

```bash
pip install -r requirements.txt
python vmc.py
```

By default, the demo saves a projected-density plot to:

```
figures/toy_diatomic_xy_density.png
```

If you want interactive display, run the demo with `show_plots=True` (see `demo()` inside `vmc.py`).

---

## Running the tests

Tests are written with `pytest` and cover:
- MH sampler sanity on a standard normal target
- τ_int and ESS consistency on an AR(1) process
- finite-difference Laplacian convergence order (2nd vs 4th)
- HO local energy constancy for the exact ground state
- end-to-end VMC energy estimate for HO ground state

```bash
pip install pytest
pytest -q
```

---

## Notes on uncertainty with MCMC

MCMC samples are correlated, so the iid standard error **σ/√N** underestimates uncertainty.  
This code estimates τ_int and uses:

- **ESS = N / (2 τ_int)**
- **SE(⟨E⟩) = √( Var(E_L) / ESS )**

where **E_L** is the local energy.

---

## Verification and numerical checks

A few built-in ideas that keep the demo “research-grade” despite being compact:

- **Finite-difference order checks**: 2nd-order vs 4th-order error scaling tests (see `tests/`)
- **Symmetry-aware plotting**: projected densities use symmetric limits and equal aspect ratio
- **Known benchmarks**: HO energies can be compared against the analytic values (E_n = n + 1/2 in chosen units)

---

## Project structure

```
.
├── vmc.py                 # single-file VMC demo 
├── requirements.txt
├── LICENSE
├── tests/
│   ├── conftest.py
│   └── test_vmc.py
└── figures/               # created at runtime
```

---

## Design choices / limitations

- Uses a simple **random-walk Metropolis** proposal
- Laplacian is computed by **finite differences** methods
- The framework assumes **real-valued ψ** in the toy systems; sign changes are handled carefully in divisions
- The “H2” example is a **toy** meant to show workflow and plotting, not a high-accuracy quantum chemistry package

---

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.
