import numpy as np

import vmc


def test_metropolis_samples_standard_normal_mean_var():
    """
    Metropolis sampler sanity check:
    sample from N(0,1) and verify mean ~ 0, var ~ 1.
    """
    def logpdf(x: np.ndarray) -> float:
        x = float(np.asarray(x).ravel()[0])
        return -0.5 * x * x  # unnormalized is fine

    cfg = vmc.MCMCConfig(
        n_samples=20_000,
        burn_in=2_000,
        thin=1,
        step_size=1.0,
        seed=123,
        target_accept=0.4,
    )

    samples, acc, info = vmc.metropolis(logpdf, x0=np.array([0.0]), cfg=cfg, adapt_steps=1000)
    x = samples[:, 0]

    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1))

    # fairly loose: random-walk MH + finite sample
    assert abs(mu) < 0.08
    assert abs(var - 1.0) < 0.12

    # acceptance should be reasonable
    assert 0.15 < acc < 0.7
    assert info["step_size_final"] > 0.0


def _ar1(phi: float, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    x = np.zeros(n, dtype=float)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def test_tau_int_ar1_reasonable():
    """
    For AR(1) with rho(t) = phi^t, the integrated autocorrelation time is:
      tau = 0.5 + sum_{t>=1} phi^t = 0.5 + phi/(1-phi)
    Your estimator uses a window rule so we test "close-ish".
    """
    phi = 0.8
    n = 40_000
    x = _ar1(phi, n, seed=1)

    tau_hat = vmc.tau_int(x, c=5.0)
    tau_true = 0.5 + phi / (1.0 - phi)  # 4.5 for phi=0.8

    # allow some bias from finite length + windowing
    assert tau_hat > 1.0
    assert abs(tau_hat - tau_true) / tau_true < 0.25  # within 25%


def test_mean_se_ess_matches_tau_int_scaling_ar1():
    """
    ESS should scale like N / (2*tau). We test that relationship is consistent.
    """
    phi = 0.7
    n = 30_000
    x = _ar1(phi, n, seed=2)

    mu, se, tau, ess = vmc.mean_se_ess(x)

    assert np.isfinite(mu)
    assert np.isfinite(se) and se > 0
    assert np.isfinite(tau) and tau >= 0.5
    assert np.isfinite(ess) and 1.0 < ess < n

    # check internal consistency: ess ~ n/(2*tau)
    ess_expected = n / (2.0 * tau)
    assert abs(ess - ess_expected) / ess_expected < 0.05  # 5%


def test_laplacian_fd_order_scaling_1d():
    """
    Check FD Laplacian convergence order on a known smooth function in 1D.
    psi(x) = exp(-a x^2)
    d2psi/dx2 = (4 a^2 x^2 - 2a) exp(-a x^2)
    """
    a = 0.7

    def psi(X, params):
        X = np.asarray(X, dtype=float)
        x = X[:, 0]
        return np.exp(-a * x * x)

    def lap_exact(X):
        X = np.asarray(X, dtype=float)
        x = X[:, 0]
        return (4.0 * a * a * x * x - 2.0 * a) * np.exp(-a * x * x)

    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, size=(200, 1))

    # 2nd order: error ~ O(h^2) => halving h reduces error by ~4
    h1, h2 = 2e-2, 1e-2
    lap1 = vmc.laplacian_fd(psi, X, None, vmc.FDConfig(h=h1, order=2))
    lap2 = vmc.laplacian_fd(psi, X, None, vmc.FDConfig(h=h2, order=2))
    err1 = float(np.mean(np.abs(lap1 - lap_exact(X))))
    err2 = float(np.mean(np.abs(lap2 - lap_exact(X))))
    assert err2 < err1
    assert (err1 / err2) > 2.5  # expect ~4, allow slack

    # 4th order: error ~ O(h^4) => halving h reduces error by ~16
    lap1_4 = vmc.laplacian_fd(psi, X, None, vmc.FDConfig(h=h1, order=4))
    lap2_4 = vmc.laplacian_fd(psi, X, None, vmc.FDConfig(h=h2, order=4))
    err1_4 = float(np.mean(np.abs(lap1_4 - lap_exact(X))))
    err2_4 = float(np.mean(np.abs(lap2_4 - lap_exact(X))))
    assert err2_4 < err1_4
    assert (err1_4 / err2_4) > 8.0  # expect ~16, allow slack


def test_local_energy_ho_ground_state_near_constant():
    """
    For the exact HO ground state, the local energy should be ~0.5 everywhere.
    We test mean and spread on a deterministic grid.
    """
    X = np.linspace(-3.0, 3.0, 121)[:, None]
    params = {"n": 0}
    fd = vmc.FDConfig(h=5e-3, order=4)

    eloc = vmc.local_energy_ho(X, params, fd=fd)
    assert np.isfinite(eloc).all()

    mean = float(np.mean(eloc))
    spread = float(np.std(eloc))

    assert abs(mean - 0.5) < 2e-3
    assert spread < 2e-2  # FD error should be small


def test_vmc_energy_pipeline_runs_and_is_reasonable_ho():
    """
    End-to-end VMC energy estimate for HO ground state:
    Should be near 0.5 with a reasonable SE and ESS.
    """
    mcmc_cfg = vmc.MCMCConfig(
        n_samples=12_000,
        burn_in=2_000,
        thin=1,
        step_size=1.0,
        seed=7,
    )
    fd = vmc.FDConfig(h=1e-2, order=4)

    res, samples, eloc = vmc.vmc_energy(
        psi=vmc.psi_ho,
        local_energy=lambda X, p: vmc.local_energy_ho(X, p, fd=fd),
        params={"n": 0},
        x0=np.array([0.0]),
        mcmc=mcmc_cfg,
    )

    assert samples.shape == (mcmc_cfg.n_samples, 1)
    assert eloc.shape == (mcmc_cfg.n_samples,)
    assert np.isfinite(eloc).all()

    assert abs(res.energy - 0.5) < 0.03
    assert np.isfinite(res.se) and res.se > 0
    assert np.isfinite(res.tau_int) and res.tau_int >= 0.5
    assert np.isfinite(res.ess) and 100.0 < res.ess < mcmc_cfg.n_samples
    assert 0.15 < res.accept_rate < 0.7