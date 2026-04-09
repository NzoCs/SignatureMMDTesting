import os
import logging
from dataclasses import dataclass, field
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ==========================================
# CONFIGURATIONS
# ==========================================


@dataclass
class PoissonComparisonConfig:
    lambda0: float = 100.0
    T: float = 10.0
    ratios: List[float] = field(
        default_factory=lambda: [
            0.5,
            0.7,
            0.8,
            0.9,
            0.95,
            1.0,
            1.05,
            1.1,
            1.2,
            1.3,
            1.5,
            2.0,
        ]
    )
    n_bank: int = 1024
    num_rep: int = 1

    def lambda1(self, ratio: float) -> float:
        return self.lambda0 * ratio


@dataclass
class HawkesOptMSEConfig:
    target_mean: float = 100.0
    beta0: float = 10.0
    T: float = 10.0
    burn_in: float = 10.0
    p0: float = 2.0
    branching_ratio_h0: float = 0.5
    alphas_h1: List[float] = field(
        default_factory=lambda: np.linspace(2, 8, 7).tolist()
    )
    n_bank: int = 128
    num_rep: int = 1

    @property
    def mu0(self) -> float:
        return self.target_mean * (1 - self.branching_ratio_h0)

    @property
    def alpha0_poly(self) -> float:
        return self.branching_ratio_h0 * self.beta0 * (self.p0 - 1)

    @property
    def mu1(self) -> float:
        return self.mu0

    def get_beta1(self, alpha1: float) -> float:
        return alpha1 / self.branching_ratio_h0


@dataclass
class HawkesKernelMSEConfig:
    target_mean: float = 100.0
    branching_ratio: float = 0.5
    beta: float = 10.0
    T: float = 10.0
    burn_in: float = 10.0
    p_values: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0, 5.0, 8.0])
    n_bank: int = 128
    num_rep: int = 1

    @property
    def mu(self) -> float:
        return self.target_mean * (1 - self.branching_ratio)

    @property
    def alpha_exp(self) -> float:
        return self.branching_ratio * self.beta

    def alpha_poly(self, p: float) -> float:
        return self.branching_ratio * self.beta * (p - 1)


@dataclass
class HawkesImprovedMSEConfig:
    target_mean: float = 100.0
    fixed_beta: float = 10.0
    alpha0: float = 5.0
    T: float = 10.0
    burn_in: float = 10.0
    alphas_h1: List[float] = field(
        default_factory=lambda: np.linspace(2, 8, 7).tolist()
    )
    n_bank: int = 128
    num_rep: int = 1

    def get_mu(self, alpha: float) -> float:
        return self.target_mean - self.fixed_beta * alpha

    @property
    def mu0(self) -> float:
        return self.get_mu(self.alpha0)


# ==========================================
# SIMULATORS & METRICS (UNIFIED)
# ==========================================


# --- Poisson ---
def sim_poisson_events(lam, num_sim, T):
    all_events = []
    for s in range(num_sim):
        events = []
        t = 0.0
        while True:
            t += np.random.exponential(1.0 / lam)
            if t >= T:
                break
            events.append(t)
        all_events.append(np.array(events))
    return all_events


def calc_rmse_poisson(events, lam_pred, lam_ref):
    mse_list = []
    mse_mean_list = []
    for ev in events:
        if len(ev) < 2:
            continue
        inter_arrival = np.diff(np.insert(ev, 0, 0.0))
        predicted_ia = 1.0 / lam_pred
        ref_ia = 1.0 / lam_ref  # dénominateur fixe = H0
        mean_ia = np.mean(inter_arrival)
        mse_list.append(np.mean(((inter_arrival - predicted_ia) / ref_ia) ** 2))
        mse_mean_list.append(np.mean(((inter_arrival - mean_ia) / ref_ia) ** 2))
    return (
        np.sqrt(np.mean(mse_list)) if mse_list else 0.0,
        np.sqrt(np.mean(mse_mean_list)) if mse_mean_list else 0.0,
    )


# --- Hawkes ---


# Kernel
def exponential_kernel(alpha, beta):
    def kernel(dt):
        return alpha * np.exp(-beta * dt)

    return kernel


# Generic Thinning
def simulate_hawkes_thinning(mu, kernel_func, end_time, start_time=0.0):
    events = []
    t = 0.0
    while t < end_time:
        lam = mu + sum(kernel_func(t - s) for s in events)
        if lam < 1e-10:
            t += 0.01
            continue
        dt = np.random.exponential(1.0 / lam)
        t += dt
        if t >= end_time:
            break
        lam_new = mu + sum(kernel_func(t - s) for s in events)
        if np.random.rand() * lam <= lam_new:
            events.append(t)
    events = np.array(events) if events else np.array([])
    if len(events) > 0:
        valid = events > start_time
        return events[valid] - start_time
    return np.array([])


# Vectorized Thinning (Poly)
def simulate_hawkes_thinning_poly(mu, alpha, beta, p, end_time, start_time=0.0):
    events = []
    t = 0.0
    while t < end_time:
        if len(events) == 0:
            lam = mu
        else:
            diffs = t - np.array(events)
            lam = mu + np.sum(alpha * (1.0 + beta * diffs) ** (-p))

        if lam < 1e-10:
            t += 0.01
            continue

        dt = np.random.exponential(1.0 / lam)
        t += dt
        if t >= end_time:
            break

        diffs_new = t - np.array(events)
        lam_new = mu + np.sum(alpha * (1.0 + beta * diffs_new) ** (-p))

        if np.random.rand() * lam <= lam_new:
            events.append(t)

    events = np.array(events)
    if len(events) > 0:
        valid = events > start_time
        return events[valid] - start_time
    return np.array([])


# Fast O(N) Simulator (Exp)
class HawkesSimulatorFast:
    def __init__(self, mu, alpha, beta, end_time, start_time=0.0):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.start_time = start_time
        self.end_time = end_time

    def simulate(self):
        times = []
        t = 0.0
        lambda_trg = 0.0

        while t < self.end_time:
            lambda_total = self.mu + lambda_trg
            dt = (
                np.random.exponential(1.0 / lambda_total)
                if lambda_total > 0
                else float("inf")
            )
            t += dt
            if t >= self.end_time:
                break

            lambda_trg *= np.exp(-self.beta * dt)
            lambda_next = self.mu + lambda_trg

            if np.random.rand() < lambda_next / lambda_total:
                times.append(t)
                lambda_trg += self.alpha

        times = np.array(times)
        valid = times > self.start_time
        return times[valid] - self.start_time


# Expected interarrival
def expected_interarrival_exp(Sn, mu, beta):
    def survivor(u):
        return np.exp(-mu * u - (Sn / beta) * (1 - np.exp(-beta * u)))

    val, _ = integrate.quad(survivor, 0, np.inf, epsabs=1e-3, epsrel=1e-3)
    return val


def expected_interarrival_poly(events_history, t_curr, mu, alpha, beta, p):
    history_shifts = 1.0 + beta * (t_curr - events_history)

    def integral_intensity(u):
        if p == 1.0:
            term = (alpha / beta) * (
                np.log(history_shifts + beta * u) - np.log(history_shifts)
            )
        else:
            term = (alpha / (beta * (1 - p))) * (
                (history_shifts + beta * u) ** (1 - p) - history_shifts ** (1 - p)
            )
        return mu * u + np.sum(term)

    def survivor(u):
        return np.exp(-integral_intensity(u))

    val, _ = integrate.quad(survivor, 0, np.inf, epsabs=1e-3, epsrel=1e-3)
    return val


def calc_rmse_hawkes_exp(
    events_list,
    mu_eval,
    alpha_eval,
    beta_eval,
    mu_ref=None,
    alpha_ref=None,
    beta_ref=None,
):
    # Si pas de ref fournie, auto-normalisation (comportement ancien)
    if mu_ref is None:
        mu_ref, alpha_ref, beta_ref = mu_eval, alpha_eval, beta_eval

    mse_all = []
    mse_mean_all = []
    for events in events_list:
        if len(events) < 5:
            continue
        squared_errors = []
        squared_errors_mean = []
        Sn_eval = 0.0
        Sn_ref = 0.0
        mean_dt = np.mean(np.diff(events))
        for i in range(1, len(events)):
            actual_dt = events[i] - events[i - 1]
            E_dt_eval = expected_interarrival_exp(Sn_eval, mu_eval, beta_eval)
            E_dt_ref = expected_interarrival_exp(Sn_ref, mu_ref, beta_ref)
            squared_errors.append(((actual_dt - E_dt_eval) / E_dt_ref) ** 2)
            squared_errors_mean.append(((actual_dt - mean_dt) / E_dt_ref) ** 2)
            Sn_eval = Sn_eval * np.exp(-beta_eval * actual_dt) + alpha_eval
            Sn_ref = Sn_ref * np.exp(-beta_ref * actual_dt) + alpha_ref
        mse_all.append(np.mean(squared_errors))
        mse_mean_all.append(np.mean(squared_errors_mean))
    return (
        np.sqrt(np.mean(mse_all)) if mse_all else 0.0,
        np.sqrt(np.mean(mse_mean_all)) if mse_mean_all else 0.0,
    )


def calc_rmse_hawkes_poly(
    events_list,
    mu_eval,
    alpha_eval,
    beta_eval,
    p_eval,
    mu_ref=None,
    alpha_ref=None,
    beta_ref=None,
    p_ref=None,
):
    if mu_ref is None:
        mu_ref, alpha_ref, beta_ref, p_ref = mu_eval, alpha_eval, beta_eval, p_eval

    mse_all = []
    mse_mean_all = []
    for events in events_list:
        if len(events) < 5:
            continue
        squared_errors = []
        squared_errors_mean = []
        mean_dt = np.mean(np.diff(events))
        for i in range(1, len(events)):
            history = events[:i]
            t_curr = events[i - 1]
            t_next = events[i]
            actual_dt = t_next - t_curr
            E_dt_eval = expected_interarrival_poly(
                history, t_curr, mu_eval, alpha_eval, beta_eval, p_eval
            )
            E_dt_ref = expected_interarrival_poly(
                history, t_curr, mu_ref, alpha_ref, beta_ref, p_ref
            )
            squared_errors.append(((actual_dt - E_dt_eval) / E_dt_ref) ** 2)
            squared_errors_mean.append(((actual_dt - mean_dt) / E_dt_ref) ** 2)
        mse_all.append(np.mean(squared_errors))
        mse_mean_all.append(np.mean(squared_errors_mean))
    return (
        np.sqrt(np.mean(mse_all)) if mse_all else 0.0,
        np.sqrt(np.mean(mse_mean_all)) if mse_mean_all else 0.0,
    )


# ==========================================
# RUNNERS FOR SUBPLOTS
# ==========================================


def run_poisson(ax):
    config = PoissonComparisonConfig()
    results_h0_pred = np.zeros((config.num_rep, len(config.ratios)))
    results_h1_pred = np.zeros((config.num_rep, len(config.ratios)))
    results_mean_pred = np.zeros((config.num_rep, len(config.ratios)))

    logging.info("Running Poisson RMSE comparison...")
    for rep in tqdm(range(config.num_rep), desc="Poisson Repetitions"):
        events_h0 = sim_poisson_events(config.lambda0, config.n_bank, config.T)
        rmse_h0, _ = calc_rmse_poisson(events_h0, config.lambda0, config.lambda0)

        for i, ratio in enumerate(config.ratios):
            lam1 = config.lambda1(ratio)
            rmse_h1, rmse_mean = calc_rmse_poisson(events_h0, lam1, config.lambda0)

            results_h0_pred[rep, i] = rmse_h0
            results_h1_pred[rep, i] = rmse_h1
            results_mean_pred[rep, i] = rmse_mean

    m_h0 = results_h0_pred.mean(axis=0)
    s_h0 = results_h0_pred.std(axis=0)
    m_h1 = results_h1_pred.mean(axis=0)
    s_h1 = results_h1_pred.std(axis=0)
    m_mean = results_mean_pred.mean(axis=0)
    s_mean = results_mean_pred.std(axis=0)

    ax.plot(config.ratios, m_h0, marker="o", label="H0 Predictor")
    ax.fill_between(config.ratios, m_h0 - s_h0, m_h0 + s_h0, alpha=0.2)
    ax.plot(config.ratios, m_h1, marker="s", label="H1 Predictor")
    ax.fill_between(config.ratios, m_h1 - s_h1, m_h1 + s_h1, alpha=0.2)
    ax.plot(config.ratios, m_mean, marker="^", label="Empirical Mean Predictor")
    ax.fill_between(config.ratios, m_mean - s_mean, m_mean + s_mean, alpha=0.2)
    ax.axvline(1.0, color="grey", linestyle="--", label="Ratio=1.0 (H1 = H0)")
    ax.set_xlabel("lambda1 / lambda0")
    ax.set_ylabel("Relative RMSE (scale-invariant)")
    ax.set_title("Poisson: H0 vs H1 Predictors")
    ax.legend()
    ax.grid(True, alpha=0.3)


def run_hawkes_opt(ax):
    config = HawkesOptMSEConfig()
    results_h0_pred = np.zeros((config.num_rep, len(config.alphas_h1)))
    results_h1_pred = np.zeros((config.num_rep, len(config.alphas_h1)))
    results_mean_pred = np.zeros((config.num_rep, len(config.alphas_h1)))

    logging.info("Running Hawkes Opt RMSE comparison...")
    for rep in tqdm(range(config.num_rep), desc="Hawkes Opt Repetitions"):
        events_h0_list = []
        for _ in range(config.n_bank):
            ev = simulate_hawkes_thinning_poly(
                config.mu0,
                config.alpha0_poly,
                config.beta0,
                config.p0,
                config.T + config.burn_in,
                config.burn_in,
            )
            events_h0_list.append(ev)

        rmse_h0, rmse_mean = calc_rmse_hawkes_poly(
            events_h0_list, config.mu0, config.alpha0_poly, config.beta0, config.p0
        )

        for i, alpha1 in enumerate(config.alphas_h1):
            beta1 = config.get_beta1(alpha1)
            mu1 = config.mu1
            rmse_h1, _ = calc_rmse_hawkes_exp(events_h0_list, mu1, alpha1, beta1)
            results_h0_pred[rep, i] = rmse_h0
            results_h1_pred[rep, i] = rmse_h1
            results_mean_pred[rep, i] = rmse_mean

    m_h0 = results_h0_pred.mean(axis=0)
    s_h0 = results_h0_pred.std(axis=0)
    m_h1 = results_h1_pred.mean(axis=0)
    s_h1 = results_h1_pred.std(axis=0)
    m_mean = results_mean_pred.mean(axis=0)
    s_mean = results_mean_pred.std(axis=0)

    ax.plot(config.alphas_h1, m_h0, marker="o", label="H0 Predictor (Power-law)")
    ax.fill_between(config.alphas_h1, m_h0 - s_h0, m_h0 + s_h0, alpha=0.2)
    ax.plot(config.alphas_h1, m_h1, marker="s", label="H1 Predictor (Exp)")
    ax.fill_between(config.alphas_h1, m_h1 - s_h1, m_h1 + s_h1, alpha=0.2)
    ax.plot(config.alphas_h1, m_mean, marker="^", label="Empirical Mean Predictor")
    ax.fill_between(config.alphas_h1, m_mean - s_mean, m_mean + s_mean, alpha=0.2)
    ax.set_xlabel("H1 base intensity alpha_1")
    ax.set_ylabel("Relative RMSE (scale-invariant)")
    ax.set_title("Hawkes (Opt): Power vs Exp")
    ax.legend()
    ax.grid(True, alpha=0.3)


def run_hawkes_kernel(ax):
    config = HawkesKernelMSEConfig()

    results_h0_pred = np.zeros((config.num_rep, len(config.p_values)))
    results_h1_pred = np.zeros((config.num_rep, len(config.p_values)))
    results_mean_pred = np.zeros((config.num_rep, len(config.p_values)))

    logging.info("Running Hawkes Kernel RMSE comparison...")
    for rep in tqdm(range(config.num_rep), desc="Hawkes Kernel Repetitions"):
        events_h0_list = []
        for _ in range(config.n_bank):
            sim = HawkesSimulatorFast(
                config.mu,
                config.alpha_exp,
                config.beta,
                config.T + config.burn_in,
                config.burn_in,
            )
            ev = sim.simulate()
            events_h0_list.append(ev)

        rmse_h0, rmse_mean = calc_rmse_hawkes_exp(
            events_h0_list, config.mu, config.alpha_exp, config.beta
        )

        for i, p_val in enumerate(config.p_values):
            alpha_p = config.alpha_poly(p_val)
            rmse_h1, _ = calc_rmse_hawkes_poly(
                events_h0_list, config.mu, alpha_p, config.beta, p_val
            )
            results_h0_pred[rep, i] = rmse_h0
            results_h1_pred[rep, i] = rmse_h1
            results_mean_pred[rep, i] = rmse_mean

    m_h0 = results_h0_pred.mean(axis=0)
    s_h0 = results_h0_pred.std(axis=0)
    m_h1 = results_h1_pred.mean(axis=0)
    s_h1 = results_h1_pred.std(axis=0)
    m_mean = results_mean_pred.mean(axis=0)
    s_mean = results_mean_pred.std(axis=0)

    ax.plot(config.p_values, m_h0, marker="o", label="H0 Predictor (Exp)")
    ax.fill_between(config.p_values, m_h0 - s_h0, m_h0 + s_h0, alpha=0.2)
    ax.plot(config.p_values, m_h1, marker="s", label="H1 Predictor (Power-law)")
    ax.fill_between(config.p_values, m_h1 - s_h1, m_h1 + s_h1, alpha=0.2)
    ax.plot(config.p_values, m_mean, marker="^", label="Empirical Mean Predictor")
    ax.fill_between(config.p_values, m_mean - s_mean, m_mean + s_mean, alpha=0.2)
    ax.set_xlabel("Power-law exponent p")
    ax.set_ylabel("Relative RMSE (scale-invariant)")
    ax.set_title("Hawkes (Kernel): Exp vs Power")
    ax.legend()
    ax.grid(True, alpha=0.3)


def run_hawkes_improved(ax):
    config = HawkesImprovedMSEConfig()
    results_h0_pred = np.zeros((config.num_rep, len(config.alphas_h1)))
    results_h1_pred = np.zeros((config.num_rep, len(config.alphas_h1)))
    results_mean_pred = np.zeros((config.num_rep, len(config.alphas_h1)))

    logging.info("Running Hawkes Improved RMSE comparison...")
    for rep in tqdm(range(config.num_rep), desc="Hawkes Improved Repetitions"):
        events_h0_list = []
        for _ in range(config.n_bank):
            sim = HawkesSimulatorFast(
                config.mu0,
                config.alpha0,
                config.fixed_beta,
                config.T + config.burn_in,
                config.burn_in,
            )
            ev = sim.simulate()
            events_h0_list.append(ev)

        rmse_h0, rmse_mean = calc_rmse_hawkes_exp(
            events_h0_list, config.mu0, config.alpha0, config.fixed_beta
        )

        for i, alpha1 in enumerate(config.alphas_h1):
            mu1 = config.get_mu(alpha1)
            rmse_h1, _ = calc_rmse_hawkes_exp(
                events_h0_list, mu1, alpha1, config.fixed_beta
            )
            results_h0_pred[rep, i] = rmse_h0
            results_h1_pred[rep, i] = rmse_h1
            results_mean_pred[rep, i] = rmse_mean

    m_h0 = results_h0_pred.mean(axis=0)
    s_h0 = results_h0_pred.std(axis=0)
    m_h1 = results_h1_pred.mean(axis=0)
    s_h1 = results_h1_pred.std(axis=0)
    m_mean = results_mean_pred.mean(axis=0)
    s_mean = results_mean_pred.std(axis=0)

    ax.plot(config.alphas_h1, m_h0, marker="o", label="H0 Predictor (Optimum)")
    ax.fill_between(config.alphas_h1, m_h0 - s_h0, m_h0 + s_h0, alpha=0.2)
    ax.plot(config.alphas_h1, m_h1, marker="s", label="H1 Predictor (Varying alpha_1)")
    ax.fill_between(config.alphas_h1, m_h1 - s_h1, m_h1 + s_h1, alpha=0.2)
    ax.plot(config.alphas_h1, m_mean, marker="^", label="Empirical Mean Predictor")
    ax.fill_between(config.alphas_h1, m_mean - s_mean, m_mean + s_mean, alpha=0.2)
    ax.axvline(
        config.alpha0,
        color="grey",
        linestyle="--",
        label=f"alpha_1={config.alpha0} (H1 = H0)",
    )
    ax.set_xlabel("H1 base intensity alpha_1")
    ax.set_ylabel("Relative RMSE (scale-invariant)")
    ax.set_title("Hawkes (Improved): Fixed beta, varying alpha")
    ax.legend()
    ax.grid(True, alpha=0.3)


# ==========================================
# MAIN EXECUTION
# ==========================================


def main():
    out_dir = "all_rmse_results"
    os.makedirs(out_dir, exist_ok=True)

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        run_poisson(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rmse_poisson.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        run_hawkes_opt(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rmse_hawkes_opt.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        run_hawkes_kernel(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rmse_hawkes_kernel.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        run_hawkes_improved(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rmse_hawkes_improved.png"), dpi=150)
        plt.close(fig)

    except KeyboardInterrupt:
        logging.info("Execution interrupted by user.")
    finally:
        logging.info("All RMSE tests finished. Plots saved.")


if __name__ == "__main__":
    main()
