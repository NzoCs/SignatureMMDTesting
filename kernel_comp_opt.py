"""
Hawkes Kernel Comparison Optimized - Signature MMD

H0: Power-law kernel (slow to simulate, simulated ONCE per rep)
    φ(t) = α0 * (1 + β0*t)^{-p0}    (p0 = 2.0, branching ratio 0.5)

H1: Exponential kernel (fast to simulate, simulated for EACH param)
    φ(t) = α1 * exp(-β1 * t)        (alpha1 sweeps from 2 to 8)

Both have the exact same target mean rate AND branching ratio (0.5).
β1 is adjusted dynamically for each α1 to maintain the constant branching ratio (β1 = α1 / 0.5).
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
    beta0: float = 10.0
    T: float = 10.0
    burn_in: float = 10.0
    grid_points: int = 300

    # H0 (Power-law) fixed parameters
    p0: float = 2.0
    branching_ratio_h0: float = 0.5
    
    # Sweep over exponential alpha1 (H1)
    alphas_h1: List[float] = field(default_factory=lambda: np.linspace(2, 8, 7).tolist())
    scalings: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 2.0])

    # Execution
    n_atoms_delta: int = 1000
    n_paths: int = 128
    n_bank: int = 1024
    num_rep: int = 10
    alpha_test: float = 0.05

    # Kernel choice
    kernel_type: str = "rbf"   # "linear" or "rbf"
    rbf_sigma: float = 1.0        # sigma for RBF kernel

    @property
    def mu0(self) -> float:
        return self.target_mean * (1 - self.branching_ratio_h0)

    @property
    def alpha0_poly(self) -> float:
        # Integral of poly(p) = alpha / (beta0 * (p-1))
        return self.branching_ratio_h0 * self.beta0 * (self.p0 - 1)

    @property
    def mu1(self) -> float:
        """Since branching ratio is identical to H0, mu1 is the same as mu0."""
        return self.mu0

    def get_beta1(self, alpha1: float) -> float:
        """Calculate dynamic beta for exponential H1 to maintain branching ratio."""
        return alpha1 / self.branching_ratio_h0

    def make_kernel(self, scaling: float = 1.0) -> SigKernel:
        if self.kernel_type == "rbf":
            static = RBFKernel(sigma=self.rbf_sigma, scaling=scaling)
        else:
            static = LinearKernel(scaling=scaling)
        return SigKernel(static_kernel=static, dyadic_order=0)


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------
def powerlaw_kernel(alpha, beta, p):
    def kernel(dt):
        return alpha * (1.0 + beta * dt) ** (-p)
    return kernel


# ---------------------------------------------------------------------------
# Fast Exponential Simulator (H1)
# ---------------------------------------------------------------------------
class HawkesSimulator:
    def __init__(self, mu, alpha, beta, dim_process, start_time=0.0, end_time=1.0, seed=None):
        self.dim_process = dim_process
        self.start_time = start_time
        self.end_time = end_time
        if seed is not None:
            np.random.seed(seed)
        self.mu = np.array(mu).reshape(dim_process)
        self.alpha = np.array(alpha).reshape(dim_process, dim_process)
        self.beta = np.array(beta).reshape(dim_process, dim_process)

    def simulate(self):
        dim = self.dim_process
        times = []
        marks = []
        t = 0.0
        lambda_trg = np.ones((dim, dim))

        while t < self.end_time:
            lambda_total = np.array([self.mu[i] + np.sum(lambda_trg[i]) for i in range(dim)])
            lambda_sum = np.sum(lambda_total)

            dt = np.random.exponential(scale=1.0 / lambda_sum) if lambda_sum > 0 else float("inf")
            t += dt

            if t >= self.end_time:
                break

            lambda_trg *= np.exp(-self.beta * dt)
            lambda_next = np.array([self.mu[i] + np.sum(lambda_trg[i]) for i in range(dim)])
            lambda_next_sum = np.sum(lambda_next)

            if np.random.rand() < lambda_next_sum / lambda_sum:
                event_dim = np.random.choice(dim, p=lambda_total / lambda_sum)
                times.append(t)
                marks.append(event_dim)
                lambda_trg[:, event_dim] += self.alpha[:, event_dim]

        times = np.array(times)
        valid = times > self.start_time
        return times[valid] - self.start_time, np.array(marks)[valid]


def sim_hawkes_model(mu, alpha, beta, num_sim, num_time_steps, T, burn_in=100.0, desc=""):
    time_grid = np.linspace(0, T, num_time_steps)
    paths = np.zeros((num_time_steps, num_sim))

    for s in tqdm(range(num_sim), desc=desc, leave=False):
        simulator = HawkesSimulator(
            mu=[mu], alpha=[[alpha]], beta=[[beta]], dim_process=1,
            start_time=burn_in, end_time=T + burn_in
        )
        event_times, _ = simulator.simulate()
        if len(event_times) > 0:
            paths[:, s] = np.searchsorted(event_times, time_grid, side="right")

    return np.concatenate((
        paths[:, :, None],
        np.repeat(time_grid[:, None, None], repeats=num_sim, axis=1)
    ), axis=2)


# ---------------------------------------------------------------------------
# Ogata Thinning Simulator (H0 Power-law)
# ---------------------------------------------------------------------------
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


def process_paths_to_tensor(h_bank, config, num_sim):
    """Convert path bank to tensor, center starting points, normalize max variance."""
    h_tensor = torch.transpose(torch.from_numpy(h_bank), 0, 1).to(device=config.device, dtype=torch.float32)

    for i in range(num_sim):
        h_tensor[i] -= h_tensor[i, 0, :]

    count_std = h_tensor[:, -1, 0].std().item()
    if count_std > 1e-8:
        h_tensor[:, :, 0] /= count_std
    if config.T > 0:
        h_tensor[:, :, 1] /= config.T

    return h_tensor


# ---------------------------------------------------------------------------
# Gram precomputation
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

    h0_sorted = np.sort(h0_dists)
    p_val_arr = (n_atoms - np.searchsorted(h0_sorted, h1_dists, side='left')) / n_atoms
    mean_pval = np.mean(p_val_arr)

    raw = (h0_dists, h1_dists, h00_dists, h01_dists)
    return 100.0 - t1e, t2e, mean_pval, raw


# ---------------------------------------------------------------------------
# Sweep & Plotting
# ---------------------------------------------------------------------------
def execute_sweep(config: KernelComparisonConfig):
    results_t1e = {s: {a: [] for a in config.alphas_h1} for s in config.scalings}
    results_t2e = {s: {a: [] for a in config.alphas_h1} for s in config.scalings}
    results_pval = {s: {a: [] for a in config.alphas_h1} for s in config.scalings}
    pooled_raw  = {s: {a: {'h0': [], 'h1': [], 'h00': [], 'h01': []}
                       for a in config.alphas_h1} for s in config.scalings}

    # Prepare H0 parameters (Power-law)
    poly_kern = powerlaw_kernel(config.alpha0_poly, config.beta0, config.p0)

    for rep in tqdm(range(config.num_rep), desc="Repetitions"):
        
        # 1. Simulate H0 ONCE per rep (since its parameters are fixed)
        h0_bank_raw = sim_hawkes_general(
            config.mu0, poly_kern, config.n_bank, config.grid_points, config.T, config.burn_in,
            desc=f"Rep {rep+1} - H0 Power-law (p={config.p0})"
        )
        h0 = process_paths_to_tensor(h0_bank_raw, config, config.n_bank)

        # 2. Iterate through H1 configs (Exponential kernel)
        for alpha1 in tqdm(config.alphas_h1, desc=f"  Rep {rep+1} — H1 alphas", leave=False):
            beta1 = config.get_beta1(alpha1)
            mu1 = config.mu1
            
            # Fast exponential simulation
            h1_bank_raw = sim_hawkes_model(
                mu1, alpha1, beta1, config.n_bank, config.grid_points, config.T, config.burn_in,
                desc=f"H1 Exp (α={alpha1:.1f}, β={beta1:.1f})"
            )
            h1 = process_paths_to_tensor(h1_bank_raw, config, config.n_bank)

            for scal in config.scalings:
                scaled_kernel = config.make_kernel(scaling=scal)

                # Precompute Grams
                K_h0   = precompute_gram_chunked(scaled_kernel, h0, h0, sym=True)
                K_h1   = precompute_gram_chunked(scaled_kernel, h1, h1, sym=True)
                K_h0h1 = precompute_gram_chunked(scaled_kernel, h0, h1, sym=False)

                t1e, t2e, mean_pval, raw = compute_errors_from_gram(
                    K_h0, K_h1, K_h0h1,
                    config.n_atoms_delta, config.n_paths, config.alpha_test)

                results_t1e[scal][alpha1].append(t1e)
                results_t2e[scal][alpha1].append(t2e)
                results_pval[scal][alpha1].append(mean_pval)

                h0_d, h1_d, h00_d, h01_d = raw
                pooled_raw[scal][alpha1]['h0'].append(h0_d)
                pooled_raw[scal][alpha1]['h1'].append(h1_d)
                pooled_raw[scal][alpha1]['h00'].append(h00_d)
                pooled_raw[scal][alpha1]['h01'].append(h01_d)

        logging.info(f"Rep {rep+1}/{config.num_rep} done.")

    # Pooled errors
    pooled_t1e = {s: {} for s in config.scalings}
    pooled_t2e = {s: {} for s in config.scalings}
    pooled_pval = {s: {} for s in config.scalings}
    for scal in config.scalings:
        for a_val in config.alphas_h1:
            a0  = np.concatenate(pooled_raw[scal][a_val]['h0'])
            a1  = np.concatenate(pooled_raw[scal][a_val]['h1'])
            a00 = np.concatenate(pooled_raw[scal][a_val]['h00'])
            a01 = np.concatenate(pooled_raw[scal][a_val]['h01'])
            n = len(a0)
            sorted_a0 = np.sort(a0)
            c1 = sorted_a0[int(n * (1 - config.alpha_test))]
            c2 = np.sort(a00)[int(n * (1 - config.alpha_test))]
            pooled_t2e[scal][a_val] = 100.0 * np.mean(a1 <= c1)
            pooled_t1e[scal][a_val] = 100.0 - 100.0 * np.mean(a01 <= c2)
            
            p_vals_pooled = (n - np.searchsorted(sorted_a0, a1, side='left')) / n
            pooled_pval[scal][a_val] = np.mean(p_vals_pooled)

    return results_t1e, results_t2e, results_pval, pooled_t1e, pooled_t2e, pooled_pval


def plot_per_rep(results_t1e, results_t2e, config, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.array(config.alphas_h1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))

    for scal, color in zip(config.scalings, colors):
        m1 = np.array([np.mean(results_t1e[scal][x]) for x in xs])
        s1 = np.array([np.std(results_t1e[scal][x]) for x in xs])
        axes[0].plot(xs, m1, label=f"Scale: {scal}", color=color, marker='o')
        axes[0].fill_between(xs, m1 - s1, m1 + s1, color=color, alpha=0.2)

        m2 = np.array([np.mean(results_t2e[scal][x]) for x in xs])
        s2 = np.array([np.std(results_t2e[scal][x]) for x in xs])
        axes[1].plot(xs, m2, label=f"Scale: {scal}", color=color, marker='o')
        axes[1].fill_between(xs, m2 - s2, m2 + s2, color=color, alpha=0.2)

    axes[0].axhline(y=5.0, color='r', linestyle='--', label='5% Target')
    axes[0].axvline(x=5.0, color='grey', linestyle=':', label="Equal branching ratio")
    axes[0].set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    axes[0].set_ylabel("P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Type 1 Error — PL vs Exp ({config.num_rep} reps)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].axvline(x=5.0, color='grey', linestyle=':', label="Equal branching ratio")
    axes[1].set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    axes[1].set_ylabel("P[Type 2 Error] (%)", fontsize=12)
    axes[1].set_title(f"Type 2 Error — PL vs Exp ({config.num_rep} reps)", fontsize=13)
    axes[1].legend(title="Scaling", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_comp_opt_per_rep.svg"), format="svg")
    plt.close()
    logging.info(f"Saved per-rep plot to {save_dir}/")


def plot_pooled(pooled_t1e, pooled_t2e, config, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.array(config.alphas_h1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))
    total_paths = config.num_rep * config.n_bank
    total_atoms = config.num_rep * config.n_atoms_delta

    for scal, color in zip(config.scalings, colors):
        pt1 = np.array([pooled_t1e[scal][x] for x in xs])
        axes[0].plot(xs, pt1, label=f"Scale: {scal}", color=color, marker='s')
        pt2 = np.array([pooled_t2e[scal][x] for x in xs])
        axes[1].plot(xs, pt2, label=f"Scale: {scal}", color=color, marker='s')

    axes[0].axhline(y=5.0, color='r', linestyle='--', label='5% Target')
    axes[0].axvline(x=5.0, color='grey', linestyle=':')
    axes[0].set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    axes[0].set_ylabel("Pooled P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Pooled: Type 1 Error ({total_paths} paths, {total_atoms} MMD)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].axvline(x=5.0, color='grey', linestyle=':')
    axes[1].set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    axes[1].set_ylabel("Pooled P[Type 2 Error] (%)", fontsize=12)
    axes[1].set_title(f"Pooled: Type 2 Error ({total_paths} paths, {total_atoms} MMD)", fontsize=13)
    axes[1].legend(title="Scaling", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_comp_opt_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved pooled plot to {save_dir}/")


def plot_pvalues(results_pval, pooled_pval, config, save_dir):
    xs = np.array(config.alphas_h1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))
    
    # 1. Per rep figure
    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        means_p = np.array([np.mean(results_pval[scal][x]) for x in xs])
        stds_p  = np.array([np.std(results_pval[scal][x]) for x in xs])
        ax.plot(xs, means_p, label=f"Scale: {scal}", color=color, marker='o')
        ax.fill_between(xs, means_p - stds_p, means_p + stds_p, color=color, alpha=0.2)
        
    ax.axvline(x=5.0, color='grey', linestyle=':', label="Equal branching ratio")
    ax.set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    ax.set_ylabel("Empirical P-value", fontsize=12)
    ax.set_title(f"Per-Rep Mean P-value ({config.num_rep} reps)", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_comp_opt_pvalues_per_rep.svg"), format="svg")
    plt.close()

    # 2. Pooled figure
    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        pt_p = np.array([pooled_pval[scal][x] for x in xs])
        ax.plot(xs, pt_p, label=f"Scale: {scal}", color=color, marker='s')
        
    ax.axvline(x=5.0, color='grey', linestyle=':', label="Equal branching ratio")
    ax.set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    ax.set_ylabel("Pooled Empirical P-value", fontsize=12)
    ax.set_title(f"Pooled P-value vs Alpha", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_comp_opt_pvalues_pooled.svg"), format="svg")
    plt.close()
    
    logging.info(f"Saved separated p-values plots to {save_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = KernelComparisonConfig()
    os.makedirs(config.data_dir, exist_ok=True)

    logging.info("Kernel Comparison: Power-Law vs Exponential")
    logging.info(f"  Target Mean = {config.target_mean} | Target Branching Ratio = {config.branching_ratio_h0}")
    logging.info(f"  H0: power-law kernel (p={config.p0}, α={config.alpha0_poly:.1f}, β0={config.beta0:.1f}, μ={config.mu0:.1f}) -> GENERATED ONCE PER REP")
    logging.info(f"  H1: exp kernel (α1 sweeps in {config.alphas_h1}, β1 adjusted dynamically, μ1={config.mu1:.1f}) -> GENERATED PER ALPHA")
    logging.info(f"  Signature kernel: {config.kernel_type}" + (f" (sigma={config.rbf_sigma})" if config.kernel_type == "rbf" else ""))

    kernel_dir = os.path.join(config.data_dir, config.kernel_type)
    os.makedirs(kernel_dir, exist_ok=True)

    results_t1e, results_t2e, results_pval, pooled_t1e, pooled_t2e, pooled_pval = execute_sweep(config)

    plot_per_rep(results_t1e, results_t2e, config, kernel_dir)
    plot_pooled(pooled_t1e, pooled_t2e, config, kernel_dir)
    plot_pvalues(results_pval, pooled_pval, config, kernel_dir)

    with open(os.path.join(kernel_dir, "metadata_opt.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    logging.info("Optimized Kernel comparison experiment finished.")


if __name__ == "__main__":
    main()
