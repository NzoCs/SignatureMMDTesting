"""
Hawkes Kernel Comparison - Signature MMD

H0: Hawkes with exponential kernel  φ(t) = α_exp * exp(-β * t)
H1: Hawkes with power-law kernel    φ(t) = α_poly * (1 + β*t)^{-p}

Both have the same branching ratio and mean rate.
Sweep over exponent p of the power-law kernel.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.mmd.mmd import SigKernel, LinearKernel, RBFKernel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class KernelComparisonConfig:
    data_dir: str = "hawkes_kernel_comparison"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Shared Hawkes parameters
    target_mean: float = 100.0
    branching_ratio: float = 0.5
    beta: float = 10.0
    T: float = 10.0
    burn_in: float = 10.0
    grid_points: int = 300

    # Sweep over power-law exponent p
    p_values: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0, 5.0, 8.0])
    scalings: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0, 4.0])

    # Execution
    n_atoms_delta: int = 1000
    n_paths: int = 256
    n_bank: int = 1024
    num_rep: int = 1
    alpha_test: float = 0.05

    # Kernel choice
    kernel_type: str = "rbf"   # "linear" or "rbf"
    rbf_sigma: float = 1.0        # sigma for RBF kernel

    @property
    def mu(self) -> float:
        return self.target_mean * (1 - self.branching_ratio)

    @property
    def alpha_exp(self) -> float:
        return self.branching_ratio * self.beta

    def alpha_poly(self, p: float) -> float:
        return self.branching_ratio * self.beta * (p - 1)

    def make_kernel(self, scaling: float = 1.0) -> SigKernel:
        if self.kernel_type == "rbf":
            static = RBFKernel(sigma=self.rbf_sigma, scaling=scaling)
        else:
            static = LinearKernel(scaling=scaling)
        return SigKernel(static_kernel=static, dyadic_order=0)


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------
def exponential_kernel(alpha, beta):
    def kernel(dt):
        return alpha * np.exp(-beta * dt)
    return kernel


def powerlaw_kernel(alpha, beta, p):
    def kernel(dt):
        return alpha * (1.0 + beta * dt) ** (-p)
    return kernel


# ---------------------------------------------------------------------------
# Hawkes simulator (general, Ogata thinning)
# ---------------------------------------------------------------------------
def simulate_hawkes_thinning(mu, kernel_func, end_time, start_time=0.0):
    """Simulate 1D Hawkes with any non-increasing kernel via Ogata thinning."""
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


def sim_hawkes_general(mu, kernel_func, num_sim, num_time_steps, T, burn_in, desc=""):
    time_grid = np.linspace(0, T, num_time_steps)
    paths = np.zeros((num_time_steps, num_sim))

    for s in tqdm(range(num_sim), desc=desc, leave=False):
        event_times = simulate_hawkes_thinning(mu, kernel_func, T + burn_in, start_time=burn_in)
        if len(event_times) > 0:
            paths[:, s] = np.searchsorted(event_times, time_grid, side="right")

    return np.concatenate((
        paths[:, :, None],
        np.repeat(time_grid[:, None, None], repeats=num_sim, axis=1)
    ), axis=2)


def load_paths(config: KernelComparisonConfig, num_sim: int, p_value: float):
    """Generate H0 (exponential) and H1 (power-law) paths."""
    exp_kern = exponential_kernel(config.alpha_exp, config.beta)
    h0_bank = sim_hawkes_general(
        config.mu, exp_kern, num_sim, config.grid_points, config.T, config.burn_in,
        desc=f"H0 exp(α={config.alpha_exp:.1f})"
    )

    a_poly = config.alpha_poly(p_value)
    poly_kern = powerlaw_kernel(a_poly, config.beta, p_value)
    h1_bank = sim_hawkes_general(
        config.mu, poly_kern, num_sim, config.grid_points, config.T, config.burn_in,
        desc=f"H1 poly(p={p_value:.1f})"
    )

    h0 = torch.transpose(torch.from_numpy(h0_bank), 0, 1).to(device=config.device, dtype=torch.float32)
    h1 = torch.transpose(torch.from_numpy(h1_bank), 0, 1).to(device=config.device, dtype=torch.float32)

    for i in range(num_sim):
        h0[i] -= h0[i, 0, :]
        h1[i] -= h1[i, 0, :]

    count_std = h0[:, -1, 0].std().item()
    if count_std > 1e-8:
        h0[:, :, 0] /= count_std
        h1[:, :, 0] /= count_std
    if config.T > 0:
        h0[:, :, 1] /= config.T
        h1[:, :, 1] /= config.T

    return h0, h1


# ---------------------------------------------------------------------------
# Gram precomputation (same as hawkes_improved_analysis.py)
# ---------------------------------------------------------------------------
def precompute_gram_chunked(sig_kernel, X, Y, sym=False, chunk_size=64):
    nx, ny = X.shape[0], Y.shape[0]
    K = torch.zeros(nx, ny, dtype=X.dtype, device=X.device)
    for i in range(0, nx, chunk_size):
        j_start = i if sym else 0
        for j in range(j_start, ny, chunk_size):
            bx = X[i:i+chunk_size]
            by = Y[j:j+chunk_size]
            with torch.no_grad():
                block = sig_kernel.compute_Gram(bx, by, sym=(sym and i == j))
            K[i:i+chunk_size, j:j+chunk_size] = block
            if sym and i != j:
                K[j:j+chunk_size, i:i+chunk_size] = block.t()
    return K


def mmd_ub_from_subgram(K_XX, K_YY, K_XY):
    nx, ny = K_XX.shape[0], K_YY.shape[0]
    xx = (K_XX.sum() - K_XX.diagonal().sum()) / (nx * (nx - 1))
    yy = (K_YY.sum() - K_YY.diagonal().sum()) / (ny * (ny - 1))
    xy = K_XY.mean()
    return (xx + yy - 2.0 * xy).item()


def compute_errors_from_gram(K_h0, K_h1, K_h0h1, n_atoms, batch_size, alpha_test):
    n0, n1 = K_h0.shape[0], K_h1.shape[0]
    K_h0, K_h1, K_h0h1 = K_h0.cpu(), K_h1.cpu(), K_h0h1.cpu()

    h0_dists  = np.empty(n_atoms)
    h1_dists  = np.empty(n_atoms)
    h00_dists = np.empty(n_atoms)
    h01_dists = np.empty(n_atoms)

    for i in range(n_atoms):
        ix1 = torch.randperm(n0)[:batch_size]
        ix2 = torch.randperm(n0)[:batch_size]
        iy  = torch.randperm(n1)[:batch_size]

        h0_dists[i] = mmd_ub_from_subgram(
            K_h0[ix1][:, ix1], K_h0[ix2][:, ix2], K_h0[ix1][:, ix2])
        h1_dists[i] = mmd_ub_from_subgram(
            K_h0[ix1][:, ix1], K_h1[iy][:, iy], K_h0h1[ix1][:, iy])

        ix3 = torch.randperm(n0)[:batch_size]
        ix4 = torch.randperm(n0)[:batch_size]
        ix5 = torch.randperm(n0)[:batch_size]

        h00_dists[i] = mmd_ub_from_subgram(
            K_h0[ix3][:, ix3], K_h0[ix4][:, ix4], K_h0[ix3][:, ix4])
        h01_dists[i] = mmd_ub_from_subgram(
            K_h0[ix3][:, ix3], K_h0[ix5][:, ix5], K_h0[ix3][:, ix5])

    crit  = np.sort(h0_dists)[int(n_atoms * (1 - alpha_test))]
    t2e   = 100.0 * np.mean(h1_dists <= crit)
    crit2 = np.sort(h00_dists)[int(n_atoms * (1 - alpha_test))]
    t1e   = 100.0 * np.mean(h01_dists <= crit2)

    raw = (h0_dists, h1_dists, h00_dists, h01_dists)
    return 100.0 - t1e, t2e, raw


# ---------------------------------------------------------------------------
# Sweep & Plotting
# ---------------------------------------------------------------------------
def execute_sweep(config: KernelComparisonConfig):
    results_t1e = {s: {p: [] for p in config.p_values} for s in config.scalings}
    results_t2e = {s: {p: [] for p in config.p_values} for s in config.scalings}
    pooled_raw  = {s: {p: {'h0': [], 'h1': [], 'h00': [], 'h01': []}
                       for p in config.p_values} for s in config.scalings}

    for rep in tqdm(range(config.num_rep), desc="Repetitions"):
        for p_val in tqdm(config.p_values, desc=f"  Rep {rep+1} — p values", leave=False):
            h0, h1 = load_paths(config, num_sim=config.n_bank, p_value=p_val)

            for scal in config.scalings:
                scaled_kernel = config.make_kernel(scaling=scal)

                K_h0   = precompute_gram_chunked(scaled_kernel, h0, h0, sym=True)
                K_h1   = precompute_gram_chunked(scaled_kernel, h1, h1, sym=True)
                K_h0h1 = precompute_gram_chunked(scaled_kernel, h0, h1, sym=False)

                t1e, t2e, raw = compute_errors_from_gram(
                    K_h0, K_h1, K_h0h1,
                    config.n_atoms_delta, config.n_paths, config.alpha_test)

                results_t1e[scal][p_val].append(t1e)
                results_t2e[scal][p_val].append(t2e)

                h0_d, h1_d, h00_d, h01_d = raw
                pooled_raw[scal][p_val]['h0'].append(h0_d)
                pooled_raw[scal][p_val]['h1'].append(h1_d)
                pooled_raw[scal][p_val]['h00'].append(h00_d)
                pooled_raw[scal][p_val]['h01'].append(h01_d)

        logging.info(f"Rep {rep+1}/{config.num_rep} done.")

    # Pooled errors
    pooled_t1e = {s: {} for s in config.scalings}
    pooled_t2e = {s: {} for s in config.scalings}
    for scal in config.scalings:
        for p_val in config.p_values:
            a0  = np.concatenate(pooled_raw[scal][p_val]['h0'])
            a1  = np.concatenate(pooled_raw[scal][p_val]['h1'])
            a00 = np.concatenate(pooled_raw[scal][p_val]['h00'])
            a01 = np.concatenate(pooled_raw[scal][p_val]['h01'])
            n = len(a0)
            c1 = np.sort(a0)[int(n * (1 - config.alpha_test))]
            c2 = np.sort(a00)[int(n * (1 - config.alpha_test))]
            pooled_t2e[scal][p_val] = 100.0 * np.mean(a1 <= c1)
            pooled_t1e[scal][p_val] = 100.0 - 100.0 * np.mean(a01 <= c2)

    return results_t1e, results_t2e, pooled_t1e, pooled_t2e


def plot_per_rep(results_t1e, results_t2e, config, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ps = np.array(config.p_values)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))

    for scal, color in zip(config.scalings, colors):
        m1 = np.array([np.mean(results_t1e[scal][p]) for p in ps])
        s1 = np.array([np.std(results_t1e[scal][p]) for p in ps])
        axes[0].plot(ps, m1, label=f"Scale: {scal}", color=color, marker='o')
        axes[0].fill_between(ps, m1 - s1, m1 + s1, color=color, alpha=0.2)

        m2 = np.array([np.mean(results_t2e[scal][p]) for p in ps])
        s2 = np.array([np.std(results_t2e[scal][p]) for p in ps])
        axes[1].plot(ps, m2, label=f"Scale: {scal}", color=color, marker='o')
        axes[1].fill_between(ps, m2 - s2, m2 + s2, color=color, alpha=0.2)

    axes[0].axhline(y=5.0, color='r', linestyle='--', label='5% Target')
    axes[0].set_xlabel("Power-law exponent p", fontsize=12)
    axes[0].set_ylabel("P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Type 1 Error — Exp vs Power-law ({config.num_rep} reps)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].set_xlabel("Power-law exponent p", fontsize=12)
    axes[1].set_ylabel("P[Type 2 Error] (%)", fontsize=12)
    axes[1].set_title(f"Type 2 Error — Exp vs Power-law ({config.num_rep} reps)", fontsize=13)
    axes[1].legend(title="Scaling", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_comparison_per_rep.svg"), format="svg")
    plt.close()
    logging.info(f"Saved per-rep plot to {save_dir}/")


def plot_pooled(pooled_t1e, pooled_t2e, config, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ps = np.array(config.p_values)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))
    total_paths = config.num_rep * config.n_bank
    total_atoms = config.num_rep * config.n_atoms_delta

    for scal, color in zip(config.scalings, colors):
        pt1 = np.array([pooled_t1e[scal][p] for p in ps])
        axes[0].plot(ps, pt1, label=f"Scale: {scal}", color=color, marker='s')
        pt2 = np.array([pooled_t2e[scal][p] for p in ps])
        axes[1].plot(ps, pt2, label=f"Scale: {scal}", color=color, marker='s')

    axes[0].axhline(y=5.0, color='r', linestyle='--', label='5% Target')
    axes[0].set_xlabel("Power-law exponent p", fontsize=12)
    axes[0].set_ylabel("Pooled P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Pooled: Type 1 Error ({total_paths} paths, {total_atoms} MMD)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].set_xlabel("Power-law exponent p", fontsize=12)
    axes[1].set_ylabel("Pooled P[Type 2 Error] (%)", fontsize=12)
    axes[1].set_title(f"Pooled: Type 2 Error ({total_paths} paths, {total_atoms} MMD)", fontsize=13)
    axes[1].legend(title="Scaling", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_comparison_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved pooled plot to {save_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = KernelComparisonConfig()
    os.makedirs(config.data_dir, exist_ok=True)

    logging.info(f"Kernel Comparison: Exp vs Power-law")
    logging.info(f"  Mean rate = {config.target_mean}, branching ratio = {config.branching_ratio}")
    logging.info(f"  H0: exp kernel (α={config.alpha_exp}, β={config.beta}), μ={config.mu}")
    logging.info(f"  H1: power-law kernel, p in {config.p_values}")
    logging.info(f"  Signature kernel: {config.kernel_type}" + (f" (sigma={config.rbf_sigma})" if config.kernel_type == "rbf" else ""))

    kernel_dir = os.path.join(config.data_dir, config.kernel_type)
    os.makedirs(kernel_dir, exist_ok=True)

    results_t1e, results_t2e, pooled_t1e, pooled_t2e = execute_sweep(config)

    plot_per_rep(results_t1e, results_t2e, config, kernel_dir)
    plot_pooled(pooled_t1e, pooled_t2e, config, kernel_dir)

    with open(os.path.join(kernel_dir, "metadata.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    logging.info("Kernel comparison experiment finished.")


if __name__ == "__main__":
    main()
