#!/usr/bin/env python3
"""
Variational Monte Carlo (VMC) mini-framework
- Generic Metropolis-Hastings for R^d using log-density
- Finite-difference Laplacian (2nd / 4th order) for local energy
- Autocorrelation-aware uncertainty estimates (ESS / tau_int)
- Gradient estimator for variational parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


Array = np.ndarray
LogPDF = Callable[[Array], float]                 # x -> log p(x)
PsiFn = Callable[[Array, Any], Array]             # X, params -> psi(X) for batch X
DLogPsiFn = Callable[[Array, Any], Array]         # X, params -> d/dparams log psi (batch)
LocalEFn = Callable[[Array, Any], Array]          # X, params -> local energy (batch)


# -----------------------------
# Config + utilities
# -----------------------------

@dataclass(frozen=True)
class MCMCConfig:
    n_samples: int = 100_000      # recorded samples
    burn_in: int = 5_000
    thin: int = 1                 # keep thin=1; use ESS instead of thinning
    step_size: float = 1.0
    seed: int = 123
    target_accept: float = 0.4    # for optional quick-tuning


@dataclass(frozen=True)
class FDConfig:
    h: float = 1e-2
    order: int = 4                # 2 or 4


def _as_2d(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X[None, :]
    return X


def _safe_divide(numer: Array, denom: Array, eps: float = 1e-300) -> Array:
    """Elementwise numer/denom with a sign-preserving floor on |denom|.

    This avoids spurious blow-ups when denom is negative or extremely close to 0.
    """
    denom = np.asarray(denom, dtype=float)
    numer = np.asarray(numer, dtype=float)
    eps = float(eps)
    s = np.sign(denom)
    s[s == 0.0] = 1.0
    denom_safe = np.where(np.abs(denom) >= eps, denom, s * eps)
    return numer / denom_safe


# -----------------------------
# Autocorrelation-aware stats
# -----------------------------

def autocorr_fft(x: Array) -> Array:
    """Unnormalized autocorrelation via FFT (O(N log N))."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 2:
        return np.array([1.0])

    # Scale to avoid overflow in FFT products when |x| is huge.
    # (Autocorrelation is scale-invariant once normalized by ac[0].)
    rms = float(np.sqrt(np.mean(x * x)))
    if not np.isfinite(rms) or rms == 0.0:
        ac = np.zeros(n, dtype=float)
        ac[0] = 1.0
        return ac
    x = x / rms
    # next power of two for speed
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=m)
    ac = np.fft.irfft(f * np.conjugate(f), n=m)[:n]
    if not np.isfinite(ac[0]) or ac[0] == 0.0:
        out = np.zeros(n, dtype=float)
        out[0] = 1.0
        return out
    ac /= ac[0]  # normalize so ac[0]=1
    return ac


def tau_int(x: Array, c: float = 5.0) -> float:
    """
    Integrated autocorrelation time with a simple windowing rule.
    tau = 0.5 + sum_{t=1..M} rho(t), stop when t > c*tau (Sokal-style).
    """
    rho = autocorr_fft(x)
    tau = 0.5
    for t in range(1, len(rho)):
        if (not np.isfinite(rho[t])) or rho[t] <= 0:
            break
        tau += rho[t]
        if t > c * tau:
            break
    return float(max(tau, 0.5))


def mean_se_ess(x: Array) -> Tuple[float, float, float, float]:
    """
    Returns (mean, se, tau_int, ess) for correlated samples.
    se = sqrt(var / ess), ess = N / (2*tau_int).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = float(np.mean(x))
    if n < 2:
        return mu, float("nan"), float("nan"), float("nan")

    # Compute variance in a scaling-safe way to avoid overflow when x is huge.
    xc = x - mu
    scale = float(np.max(np.abs(xc)))
    if not np.isfinite(scale) or scale == 0.0:
        var = 0.0
    else:
        y = xc / scale
        var = float((scale * scale) * (np.sum(y * y) / (n - 1)))

    tau = tau_int(x)
    ess = n / (2.0 * tau)
    se = np.sqrt(var / ess)
    return mu, float(se), float(tau), float(ess)


# -----------------------------
# Metropolis-Hastings (generic)
# -----------------------------

def metropolis(
    logpdf: LogPDF,
    x0: Array,
    cfg: MCMCConfig,
    adapt_steps: int = 2_000,
) -> Tuple[Array, float, Dict[str, float]]:
    """
    Random-walk Metropolis with Gaussian proposal.
    Uses logpdf for numerical stability.

    Returns:
      samples: (cfg.n_samples, d)
      accept_rate: float
      info: dict with final step size, burn-in accept rate, etc.
    """
    rng = np.random.default_rng(cfg.seed)
    x = np.asarray(x0, dtype=float).copy()
    d = x.size

    step = float(cfg.step_size)
    logp = float(logpdf(x))
    n_acc = 0
    n_tot = 0
    burn_acc = 0
    burn_tot = 0

    for t in range(adapt_steps):
        x_prop = x + step * rng.normal(size=d)
        logp_prop = float(logpdf(x_prop))
        if np.log(rng.uniform()) < (logp_prop - logp):
            x, logp = x_prop, logp_prop
            burn_acc += 1
        burn_tot += 1

        # adjust every 100 steps
        if (t + 1) % 100 == 0:
            acc = burn_acc / max(burn_tot, 1)
            if acc < cfg.target_accept:
                step *= 0.9
            else:
                step *= 1.1
            step = max(step, 1e-6)

    # --- production sampling ---
    samples = []
    while len(samples) < cfg.n_samples:
        x_prop = x + step * rng.normal(size=d)
        logp_prop = float(logpdf(x_prop))
        if np.log(rng.uniform()) < (logp_prop - logp):
            x, logp = x_prop, logp_prop
            n_acc += 1
        n_tot += 1

        if n_tot > cfg.burn_in and (n_tot - cfg.burn_in) % cfg.thin == 0:
            samples.append(x.copy())

    samples = np.asarray(samples)
    info = {
        "step_size_final": step,
        "burn_accept": burn_acc / max(burn_tot, 1),
        "total_accept": n_acc / max(n_tot, 1),
    }
    return samples, info["total_accept"], info


# -----------------------------
# Finite-difference Laplacian
# -----------------------------

def laplacian_fd(psi: PsiFn, X: Array, params: Any, fd: FDConfig) -> Array:
    """
    Compute ∇² psi(X) in R^d using central finite differences.
    X: (N, d)
    returns: (N,)
    """
    X = _as_2d(X)
    n, d = X.shape
    h = float(fd.h)

    psi0 = psi(X, params)  # (N,)
    lap = np.zeros(n, dtype=float)

    if fd.order == 2:
        for j in range(d):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, j] += h
            Xm[:, j] -= h
            lap += (psi(Xp, params) - 2.0 * psi0 + psi(Xm, params)) / (h * h)

    elif fd.order == 4:
        for j in range(d):
            Xp1 = X.copy(); Xm1 = X.copy()
            Xp2 = X.copy(); Xm2 = X.copy()
            Xp1[:, j] += h;  Xm1[:, j] -= h
            Xp2[:, j] += 2*h; Xm2[:, j] -= 2*h
            lap += (
                -psi(Xp2, params)
                + 16.0 * psi(Xp1, params)
                - 30.0 * psi0
                + 16.0 * psi(Xm1, params)
                - psi(Xm2, params)
            ) / (12.0 * h * h)
    else:
        raise ValueError("fd.order must be 2 or 4")

    return lap


# -----------------------------
# VMC estimators
# -----------------------------

@dataclass
class VMCResult:
    energy: float
    se: float
    tau_int: float
    ess: float
    accept_rate: float
    sampler_info: Dict[str, float]


def vmc_energy(
    psi: PsiFn,
    local_energy: LocalEFn,
    params: Any,
    x0: Array,
    mcmc: MCMCConfig,
) -> Tuple[VMCResult, Array, Array]:
    """
    Sample from |psi|^2 and estimate <H> with autocorrelation-aware SE.
    Returns (result, samples, local_energies).
    """
    def logpdf(x: Array) -> float:
        # log |psi(x)|^2 = 2 log |psi|
        # Expect psi positive real for these toy systems; use abs for safety.
        val = float(np.abs(psi(_as_2d(x), params)[0]))
        return 2.0 * np.log(max(val, 1e-300))

    samples, acc, info = metropolis(logpdf, x0=x0, cfg=mcmc)
    eloc = local_energy(samples, params)

    e_mean, e_se, tau, ess = mean_se_ess(eloc)
    res = VMCResult(
        energy=e_mean,
        se=e_se,
        tau_int=tau,
        ess=ess,
        accept_rate=acc,
        sampler_info=info,
    )
    return res, samples, eloc


def vmc_energy_and_grad(
    psi: PsiFn,
    local_energy: LocalEFn,
    dlogpsi: DLogPsiFn,
    params: Any,
    x0: Array,
    mcmc: MCMCConfig,
) -> Tuple[VMCResult, Array]:
    """
    Gradient estimator:
      d/dθ <H> = 2 * < (E_L - <E_L>) * d/dθ log psi >
    Works for scalar or vector params (dlogpsi returns (N,p)).
    """
    res, samples, eloc = vmc_energy(psi, local_energy, params, x0, mcmc)
    O = dlogpsi(samples, params)  # (N,) or (N,p)
    el = eloc - np.mean(eloc)

    if O.ndim == 1:
        grad = 2.0 * float(np.mean(el * O))
    else:
        grad = 2.0 * np.mean(el[:, None] * O, axis=0)
    return res, np.asarray(grad)


def gradient_descent(
    psi: PsiFn,
    local_energy: LocalEFn,
    dlogpsi: DLogPsiFn,
    theta0: Array,
    x0: Array,
    mcmc: MCMCConfig,
    n_iter: int = 30,
    lr: float = 0.1,
    clamp: Optional[Callable[[Array], Array]] = None,
) -> Tuple[Array, Array]:
    """
    Basic stochastic gradient descent for variational parameters.
    Returns (history, best_theta).
    history rows: [iter, E, SE, tau, ESS, accept, theta..., grad...]
    """
    theta = np.asarray(theta0, dtype=float).copy()
    hist = []

    for k in range(n_iter):
        res, grad = vmc_energy_and_grad(psi, local_energy, dlogpsi, theta, x0, mcmc)
        theta = theta - lr * grad
        if clamp is not None:
            theta = clamp(theta)

        row = np.concatenate([
            np.array([k, res.energy, res.se, res.tau_int, res.ess, res.accept_rate], dtype=float),
            theta.ravel().astype(float),
            np.atleast_1d(grad).ravel().astype(float),
        ])
        hist.append(row)

        print(
            f"iter={k:02d}  E={res.energy:+.6f} ± {res.se:.2e}  "
            f"tau={res.tau_int:.2f}  ESS={res.ess:.0f}  acc={res.accept_rate:.3f}  "
            f"theta={theta}"
        )

    hist = np.vstack(hist)
    best_idx = int(np.argmin(hist[:, 1]))
    best_theta = hist[best_idx, 6:6+theta.size]
    return hist, best_theta


# -----------------------------
# Plotting helpers (symmetry-aware)
# -----------------------------

def symmetric_limits(samples_xy: Array, q: float = 0.995) -> float:
    """Choose symmetric plot half-range L based on a high quantile of |x| and |y|."""
    x = samples_xy[:, 0]
    y = samples_xy[:, 1]
    L = float(np.quantile(np.maximum(np.abs(x), np.abs(y)), q))
    return max(L, 1e-6)


def plot_xy_density(samples: Array, title: str, bins: int = 200, q: float = 0.995):
    """2D density with enforced equal aspect + symmetric limits."""
    xy = samples[:, :2]
    L = symmetric_limits(xy, q=q)

    plt.figure(figsize=(6, 5))
    plt.hist2d(xy[:, 0], xy[:, 1], bins=bins, range=[[-L, L], [-L, L]], density=True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.colorbar(label="Probability density")
    plt.tight_layout()


# -----------------------------
# Example systems
# -----------------------------

# --- 1D Harmonic Oscillator ---
def hermite(x: Array, n: int) -> Array:
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    Hm1 = np.ones_like(x)
    H0 = 2.0 * x
    for k in range(1, n):
        Hp1 = 2.0 * x * H0 - 2.0 * k * Hm1
        Hm1, H0 = H0, Hp1
    return H0


def psi_ho(X: Array, params: Dict[str, Any]) -> Array:
    X = _as_2d(X)
    x = X[:, 0]
    n = int(params.get("n", 0))
    return hermite(x, n) * np.exp(-0.5 * x * x)


def local_energy_ho(X: Array, params: Dict[str, Any], fd: FDConfig = FDConfig()) -> Array:
    X = _as_2d(X)
    x = X[:, 0]
    # use 1D laplacian_fd on a 1D embedding
    lap = laplacian_fd(psi_ho, X, params, fd)
    psi0 = psi_ho(X, params)
    # For excited states psi0 changes sign; using np.maximum would clamp negatives
    # to eps and create huge spurious energies. Use sign-preserving safe division.
    KE = -0.5 * _safe_divide(lap, psi0, eps=1e-300)
    PE = 0.5 * x * x
    return KE + PE


# --- 3D Hydrogen (1s trial) ---
def psi_h_atom(X: Array, theta: Array) -> Array:
    X = _as_2d(X)
    r = np.linalg.norm(X, axis=1)
    th = float(np.asarray(theta).ravel()[0])
    return np.exp(-th * r)


def dlogpsi_h_atom(X: Array, theta: Array) -> Array:
    X = _as_2d(X)
    r = np.linalg.norm(X, axis=1)
    return -r  # d/dtheta log psi = -r


def local_energy_h_atom(X: Array, theta: Array, fd: FDConfig = FDConfig()) -> Array:
    X = _as_2d(X)
    r = np.linalg.norm(X, axis=1)
    lap = laplacian_fd(psi_h_atom, X, theta, fd)
    psi0 = psi_h_atom(X, theta)
    KE = -0.5 * _safe_divide(lap, psi0, eps=1e-300)
    V = -1.0 / np.maximum(r, 1e-12)
    return KE + V


# --- 6D H2 (two electrons, fixed nuclei on x-axis) ---
def nuclear_positions(Q: float) -> Tuple[Array, Array]:
    q1 = np.array([-0.5 * Q, 0.0, 0.0], dtype=float)
    q2 = np.array([+0.5 * Q, 0.0, 0.0], dtype=float)
    return q1, q2


def psi_h2(X: Array, params: Dict[str, Any]) -> Array:
    """
    X: (N,6) = (r1x,r1y,r1z,r2x,r2y,r2z)
    params: {"theta": (t1,t2,t3), "Q": float}
    """
    X = _as_2d(X)
    th = np.asarray(params["theta"], dtype=float).ravel()
    t1, t2, t3 = th
    Q = float(params["Q"])
    q1, q2 = nuclear_positions(Q)

    r1 = X[:, 0:3]
    r2 = X[:, 3:6]

    r1A = np.linalg.norm(r1 - q1, axis=1)
    r1B = np.linalg.norm(r1 - q2, axis=1)
    r2A = np.linalg.norm(r2 - q1, axis=1)
    r2B = np.linalg.norm(r2 - q2, axis=1)
    r12 = np.linalg.norm(r1 - r2, axis=1)

    phi_AB = np.exp(-t1 * (r1A + r2B))
    phi_BA = np.exp(-t1 * (r1B + r2A))
    J = np.exp(-t2 / (1.0 + t3 * r12))
    return (phi_AB + phi_BA) * J


def dlogpsi_h2(X: Array, params: Dict[str, Any]) -> Array:
    X = _as_2d(X)
    th = np.asarray(params["theta"], dtype=float).ravel()
    t1, t2, t3 = th
    Q = float(params["Q"])
    q1, q2 = nuclear_positions(Q)

    r1 = X[:, 0:3]
    r2 = X[:, 3:6]

    r1A = np.linalg.norm(r1 - q1, axis=1)
    r1B = np.linalg.norm(r1 - q2, axis=1)
    r2A = np.linalg.norm(r2 - q1, axis=1)
    r2B = np.linalg.norm(r2 - q2, axis=1)
    r12 = np.linalg.norm(r1 - r2, axis=1)

    phi_AB = np.exp(-t1 * (r1A + r2B))
    phi_BA = np.exp(-t1 * (r1B + r2A))
    denom = np.maximum(phi_AB + phi_BA, 1e-300)

    dlog_t1 = (-(r1A + r2B) * phi_AB - (r1B + r2A) * phi_BA) / denom

    f = np.maximum(1.0 + t3 * r12, 1e-12)
    dlog_t2 = -1.0 / f
    dlog_t3 = t2 * r12 / (f * f)

    return np.column_stack([dlog_t1, dlog_t2, dlog_t3])


def local_energy_h2(X: Array, params: Dict[str, Any], fd: FDConfig = FDConfig(order=2, h=1e-2)) -> Array:
    """
    Hamiltonian (dimensionless):
      H = -1/2 (∇1^2 + ∇2^2) - Σ_i Σ_A 1/|ri-qA| + 1/|r1-r2| + 1/|q1-q2|
    """
    X = _as_2d(X)
    Q = float(params["Q"])
    q1, q2 = nuclear_positions(Q)

    psi0 = psi_h2(X, params)
    lap = laplacian_fd(psi_h2, X, params, fd)
    KE = -0.5 * _safe_divide(lap, psi0, eps=1e-300)

    r1 = X[:, 0:3]
    r2 = X[:, 3:6]
    r1A = np.linalg.norm(r1 - q1, axis=1)
    r1B = np.linalg.norm(r1 - q2, axis=1)
    r2A = np.linalg.norm(r2 - q1, axis=1)
    r2B = np.linalg.norm(r2 - q2, axis=1)
    r12 = np.linalg.norm(r1 - r2, axis=1)

    eps = 1e-12
    V_en = -(1/np.maximum(r1A, eps) + 1/np.maximum(r1B, eps) + 1/np.maximum(r2A, eps) + 1/np.maximum(r2B, eps))
    V_ee = 1.0 / np.maximum(r12, eps)
    V_nn = 1.0 / max(Q, eps)
    return KE + (V_en + V_ee + V_nn)


# -----------------------------
# Demo runner
# -----------------------------

def demo(show_plots: bool = False, outdir: str = "figures"):
    """
    Quick usage examples for the VMC mini-framework.

    These are intentionally lightweight "toy-system" demos meant to illustrate:
    - sampling from |psi|^2 with Metropolis–Hastings
    - estimating <H> with autocorrelation-aware SE (ESS / tau_int)
    - optimising a simple variational parameter via stochastic gradient descent
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    print("\nVMC mini-framework: quick toy demos")
    print("------------------------------------------------------------")

    # 1) Harmonic oscillator: one or two states only (quick sanity check)
    print("\n[Demo 1] 1D harmonic oscillator — energy estimate")
    mcmc_ho = MCMCConfig(
        n_samples=25_000, burn_in=3_000, thin=1,
        step_size=1.0, seed=1
    )
    fd_ho = FDConfig(h=1e-2, order=4)

    for n in (0, 2):
        params = {"n": n}
        res, samples, eloc = vmc_energy(
            psi=psi_ho,
            local_energy=lambda X, p: local_energy_ho(X, p, fd=fd_ho),
            params=params,
            x0=np.array([0.0]),
            mcmc=mcmc_ho,
        )

        e_exact = n + 0.5
        print(
            f"  n={n}:  E={res.energy:.8f} ± {res.se:.2e}  "
            f"(exact {e_exact:.3f})  tau={res.tau_int:.2f}  ESS={res.ess:.0f}  acc={res.accept_rate:.3f}"
        )

    # 2) Hydrogen: optimise a single variational parameter theta
    print("\n[Demo 2] 3D hydrogen-like trial wavefunction — optimise θ")
    mcmc_h = MCMCConfig(
        n_samples=30_000, burn_in=4_000, thin=1,
        step_size=0.8, seed=2
    )
    fd_h = FDConfig(h=1e-2, order=4)

    hist, best = gradient_descent(
        psi=psi_h_atom,
        local_energy=lambda X, th: local_energy_h_atom(X, th, fd=fd_h),
        dlogpsi=lambda X, th: dlogpsi_h_atom(X, th),
        theta0=np.array([0.7]),
        x0=np.zeros(3),
        mcmc=mcmc_h,
        n_iter=12,
        lr=0.12,
        clamp=lambda th: np.maximum(th, 0.05),
    )
    best_theta = float(np.asarray(best).ravel()[0])
    print(f"  best θ (from run) ≈ {best_theta:.5f}")

    # 3) Two-electron diatomic toy: one-shot density plot (optional)
    print("\n[Demo 3] Two-electron diatomic toy model — density snapshot")
    mcmc_h2 = MCMCConfig(
        n_samples=20_000, burn_in=6_000, thin=1,
        step_size=0.6, seed=3
    )
    fd_h2 = FDConfig(h=1e-2, order=2)  # speed over accuracy for demo

    params = {"theta": np.array([1.10, 0.12, 0.10]), "Q": 1.4}
    res, samples, _ = vmc_energy(
        psi=psi_h2,
        local_energy=lambda X, p: local_energy_h2(X, p, fd=fd_h2),
        params=params,
        x0=np.zeros(6),
        mcmc=mcmc_h2,
    )
    print(
        f"  E={res.energy:+.6f} ± {res.se:.2e}  tau={res.tau_int:.2f}  ESS={res.ess:.0f}  acc={res.accept_rate:.3f}"
    )

    # Projected xy density (both electrons) — save by default
    r1 = samples[:, 0:3]
    r2 = samples[:, 3:6]
    xy = np.vstack([r1[:, :2], r2[:, :2]])

    plot_xy_density(
        xy,
        title="Two-electron diatomic toy: projected density (x–y)",
        bins=220,
        q=0.995
    )
    fig_path = os.path.join(outdir, "toy_diatomic_xy_density.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"  saved plot -> {fig_path}")

    if show_plots:
        # If you really want interactive plots, re-run with show_plots=True
        img = plt.imread(fig_path)
        plt.figure(figsize=(6, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    demo(show_plots=False)
