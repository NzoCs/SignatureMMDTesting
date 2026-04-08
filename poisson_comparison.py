"""
Poisson Process Comparison - Signature MMD

H0: Poisson process with rate λ0
H1: Poisson process with rate λ1
Sweep over the ratio λ1/λ0.
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
class PoissonComparisonConfig:
    data_dir: str = "poisson_comparison"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Poisson parameters
    lambda0: float = 100.0  # H0 rate
    T: float = 1.0
    grid_points: int = 100

    # Sweep over ratio λ1/λ0
    ratios: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0])
    scalings: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0])

    # Execution
    n_atoms_delta: int = 500
    n_paths: int = 128
    n_bank: int = 512
    num_rep: int = 10
    alpha_test: float = 0.05

    # Kernel choice
    kernel_type: str = "linear"   # "linear" or "rbf"
    rbf_sigma: float = 1.0        # sigma for RBF kernel

    def lambda1(self, ratio: float) -> float:
        return self.lambda0 * ratio

    def make_kernel(self, scaling: float = 1.0) -> SigKernel:
        if self.kernel_type == "rbf":
            static = RBFKernel(sigma=self.rbf_sigma, scaling=scaling)
        else:
            static = LinearKernel(scaling=scaling)
        return SigKernel(static_kernel=static, dyadic_order=0)


# ---------------------------------------------------------------------------
# Poisson simulator
# ---------------------------------------------------------------------------
def sim_poisson(lam, num_sim, num_time_steps, T):
    """Simulate homogeneous Poisson counting processes on [0, T]."""
    time_grid = np.linspace(0, T, num_time_steps)
    paths = np.zeros((num_time_steps, num_sim))

    for s in range(num_sim):
        # Generate events via exponential inter-arrivals
        events = []
        t = 0.0
        while True:
            t += np.random.exponential(1.0 / lam)
            if t >= T:
                break
            events.append(t)
        if len(events) > 0:
            paths[:, s] = np.searchsorted(np.array(events), time_grid, side="right")

    return np.concatenate((
        paths[:, :, None],
        np.repeat(time_grid[:, None, None], repeats=num_sim, axis=1)
    ), axis=2)


def load_poisson_paths(config: PoissonComparisonConfig, num_sim: int, ratio: float):
    """Generate H0 and H1 Poisson paths."""
    h0_bank = sim_poisson(config.lambda0, num_sim, config.grid_points, config.T)

    lam1 = config.lambda1(ratio)
    h1_bank = sim_poisson(lam1, num_sim, config.grid_points, config.T)

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
# Gram precomputation
# ---------------------------------------------------------------------------
def precompute_gram_chunked(sig_kernel, X, Y, sym=False, chunk_size=128):
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
def execute_sweep(config: PoissonComparisonConfig):
    results_t1e = {s: {r: [] for r in config.ratios} for s in config.scalings}
    results_t2e = {s: {r: [] for r in config.ratios} for s in config.scalings}
    results_pval = {s: {r: [] for r in config.ratios} for s in config.scalings}
    pooled_raw  = {s: {r: {'h0': [], 'h1': [], 'h00': [], 'h01': []}
                       for r in config.ratios} for s in config.scalings}

    for rep in tqdm(range(config.num_rep), desc="Repetitions"):
        for ratio in tqdm(config.ratios, desc=f"  Rep {rep+1} — ratios", leave=False):
            h0, h1 = load_poisson_paths(config, num_sim=config.n_bank, ratio=ratio)

            for scal in config.scalings:
                scaled_kernel = config.make_kernel(scaling=scal)

                K_h0   = precompute_gram_chunked(scaled_kernel, h0, h0, sym=True)
                K_h1   = precompute_gram_chunked(scaled_kernel, h1, h1, sym=True)
                K_h0h1 = precompute_gram_chunked(scaled_kernel, h0, h1, sym=False)

                t1e, t2e, mean_pval, raw = compute_errors_from_gram(
                    K_h0, K_h1, K_h0h1,
                    config.n_atoms_delta, config.n_paths, config.alpha_test)

                results_t1e[scal][ratio].append(t1e)
                results_t2e[scal][ratio].append(t2e)
                results_pval[scal][ratio].append(mean_pval)

                h0_d, h1_d, h00_d, h01_d = raw
                pooled_raw[scal][ratio]['h0'].append(h0_d)
                pooled_raw[scal][ratio]['h1'].append(h1_d)
                pooled_raw[scal][ratio]['h00'].append(h00_d)
                pooled_raw[scal][ratio]['h01'].append(h01_d)

        logging.info(f"Rep {rep+1}/{config.num_rep} done.")

    # Pooled errors
    pooled_t1e = {s: {} for s in config.scalings}
    pooled_t2e = {s: {} for s in config.scalings}
    pooled_pval = {s: {} for s in config.scalings}
    for scal in config.scalings:
        for ratio in config.ratios:
            a0  = np.concatenate(pooled_raw[scal][ratio]['h0'])
            a1  = np.concatenate(pooled_raw[scal][ratio]['h1'])
            a00 = np.concatenate(pooled_raw[scal][ratio]['h00'])
            a01 = np.concatenate(pooled_raw[scal][ratio]['h01'])
            n = len(a0)
            sorted_a0 = np.sort(a0)
            c1 = sorted_a0[int(n * (1 - config.alpha_test))]
            c2 = np.sort(a00)[int(n * (1 - config.alpha_test))]
            pooled_t2e[scal][ratio] = 100.0 * np.mean(a1 <= c1)
            pooled_t1e[scal][ratio] = 100.0 - 100.0 * np.mean(a01 <= c2)
            
            p_vals_pooled = (n - np.searchsorted(sorted_a0, a1, side='left')) / n
            pooled_pval[scal][ratio] = np.mean(p_vals_pooled)

    return results_t1e, results_t2e, results_pval, pooled_t1e, pooled_t2e, pooled_pval


def plot_per_rep(results_t1e, results_t2e, config, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rs = np.array(config.ratios)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))

    for scal, color in zip(config.scalings, colors):
        m1 = np.array([np.mean(results_t1e[scal][r]) for r in rs])
        s1 = np.array([np.std(results_t1e[scal][r]) for r in rs])
        axes[0].plot(rs, m1, label=f"Scale: {scal}", color=color, marker='o')
        axes[0].fill_between(rs, m1 - s1, m1 + s1, color=color, alpha=0.2)

        m2 = np.array([np.mean(results_t2e[scal][r]) for r in rs])
        s2 = np.array([np.std(results_t2e[scal][r]) for r in rs])
        axes[1].plot(rs, m2, label=f"Scale: {scal}", color=color, marker='o')
        axes[1].fill_between(rs, m2 - s2, m2 + s2, color=color, alpha=0.2)

    axes[0].axhline(y=5.0, color='r', linestyle='--', label='5% Target')
    axes[0].axvline(x=1.0, color='grey', linestyle=':', alpha=0.5, label=r'$\lambda_1/\lambda_0=1$')
    axes[0].set_xlabel(r"$\lambda_1 / \lambda_0$", fontsize=12)
    axes[0].set_ylabel("P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Type 1 Error — Poisson ({config.num_rep} reps, {config.n_bank} paths/rep)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=9)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].axvline(x=1.0, color='grey', linestyle=':', alpha=0.5, label=r'$\lambda_1/\lambda_0=1$')
    axes[1].set_xlabel(r"$\lambda_1 / \lambda_0$", fontsize=12)
    axes[1].set_ylabel("P[Type 2 Error] (%)", fontsize=12)
    axes[1].set_title(f"Type 2 Error — Poisson ({config.num_rep} reps, {config.n_bank} paths/rep)", fontsize=13)
    axes[1].legend(title="Scaling", fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "poisson_errors_per_rep.svg"), format="svg")
    plt.close()
    logging.info(f"Saved per-rep plot to {save_dir}/")


def plot_pooled(pooled_t1e, pooled_t2e, config, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    rs = np.array(config.ratios)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))
    total_paths = config.num_rep * config.n_bank
    total_atoms = config.num_rep * config.n_atoms_delta

    for scal, color in zip(config.scalings, colors):
        pt1 = np.array([pooled_t1e[scal][r] for r in rs])
        axes[0].plot(rs, pt1, label=f"Scale: {scal}", color=color, marker='s')
        pt2 = np.array([pooled_t2e[scal][r] for r in rs])
        axes[1].plot(rs, pt2, label=f"Scale: {scal}", color=color, marker='s')

    axes[0].axhline(y=5.0, color='r', linestyle='--', label='5% Target')
    axes[0].axvline(x=1.0, color='grey', linestyle=':', alpha=0.5)
    axes[0].set_xlabel(r"$\lambda_1 / \lambda_0$", fontsize=12)
    axes[0].set_ylabel("Pooled P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Pooled: Type 1 Error ({total_paths} paths, {total_atoms} MMD)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=9)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].axvline(x=1.0, color='grey', linestyle=':', alpha=0.5)
    axes[1].set_xlabel(r"$\lambda_1 / \lambda_0$", fontsize=12)
    axes[1].set_ylabel("Pooled P[Type 2 Error] (%)", fontsize=12)
    axes[1].set_title(f"Pooled: Type 2 Error ({total_paths} paths, {total_atoms} MMD)", fontsize=13)
    axes[1].legend(title="Scaling", fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "poisson_errors_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved pooled plot to {save_dir}/")


def plot_pvalues(results_pval, pooled_pval, config, save_dir):
    rs = np.array(config.ratios)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))
    
    # 1. Per rep figure
    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        means_p = np.array([np.mean(results_pval[scal][r]) for r in rs])
        stds_p  = np.array([np.std(results_pval[scal][r]) for r in rs])
        ax.plot(rs, means_p, label=f"Scale: {scal}", color=color, marker='o')
        ax.fill_between(rs, means_p - stds_p, means_p + stds_p, color=color, alpha=0.2)
        
    ax.axvline(x=1.0, color='grey', linestyle=':', alpha=0.5, label=r'$\lambda_1/\lambda_0=1$')
    ax.set_xlabel(r"$\lambda_1 / \lambda_0$", fontsize=12)
    ax.set_ylabel("Empirical P-value", fontsize=12)
    ax.set_title(f"Per-Rep Mean P-value ({config.num_rep} reps)", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "poisson_comparison_pvalues_per_rep.svg"), format="svg")
    plt.close()

    # 2. Pooled figure
    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        pt_p = np.array([pooled_pval[scal][r] for r in rs])
        ax.plot(rs, pt_p, label=f"Scale: {scal}", color=color, marker='s')
        
    ax.axvline(x=1.0, color='grey', linestyle=':', alpha=0.5, label=r'$\lambda_1/\lambda_0=1$')
    ax.set_xlabel(r"$\lambda_1 / \lambda_0$", fontsize=12)
    ax.set_ylabel("Pooled Empirical P-value", fontsize=12)
    ax.set_title(f"Pooled P-value vs Ratio", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "poisson_comparison_pvalues_pooled.svg"), format="svg")
    plt.close()
    
    logging.info(f"Saved separated p-values plots to {save_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    config = PoissonComparisonConfig()
    os.makedirs(config.data_dir, exist_ok=True)

    logging.info(f"Poisson Comparison: λ0={config.lambda0}, ratios={config.ratios}")
    logging.info(f"Signature kernel: {config.kernel_type}" + (f" (sigma={config.rbf_sigma})" if config.kernel_type == "rbf" else ""))

    kernel_dir = os.path.join(config.data_dir, config.kernel_type)
    os.makedirs(kernel_dir, exist_ok=True)

    results_t1e, results_t2e, results_pval, pooled_t1e, pooled_t2e, pooled_pval = execute_sweep(config)

    plot_per_rep(results_t1e, results_t2e, config, kernel_dir)
    plot_pooled(pooled_t1e, pooled_t2e, config, kernel_dir)
    plot_pvalues(results_pval, pooled_pval, config, kernel_dir)

    with open(os.path.join(kernel_dir, "metadata.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    logging.info("Poisson comparison experiment finished.")


if __name__ == "__main__":
    main()
