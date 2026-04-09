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

from experiment_utils import (
    precompute_gram_chunked, compute_errors_from_gram,
    process_paths_pair_to_tensor,
    init_results_dicts, accumulate_results, compute_pooled_stats,
    make_sig_kernel,
)

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
    scalings: List[float] = field(default_factory=lambda: [1.0])

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

    def make_kernel(self, scaling: float = 1.0):
        return make_sig_kernel(self.kernel_type, self.rbf_sigma, scaling)


# ---------------------------------------------------------------------------
# Poisson simulator
# ---------------------------------------------------------------------------
def sim_poisson(lam, num_sim, num_time_steps, T):
    """Simulate homogeneous Poisson counting processes on [0, T]."""
    time_grid = np.linspace(0, T, num_time_steps)
    paths = np.zeros((num_time_steps, num_sim))

    for s in range(num_sim):
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


def load_poisson_paths(config, num_sim, ratio):
    """Generate H0 and H1 Poisson paths."""
    h0_bank = sim_poisson(config.lambda0, num_sim, config.grid_points, config.T)
    h1_bank = sim_poisson(config.lambda1(ratio), num_sim, config.grid_points, config.T)
    return process_paths_pair_to_tensor(h0_bank, h1_bank, config, num_sim)


# ---------------------------------------------------------------------------
# Sweep & Plotting
# ---------------------------------------------------------------------------
def execute_sweep(config):
    results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_raw = \
        init_results_dicts(config.scalings, config.ratios)

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

                accumulate_results(results_t1e, results_t2e, results_pval,
                                   results_norm_mmd, pooled_raw,
                                   scal, ratio, t1e, t2e, mean_pval, raw)

        logging.info(f"Rep {rep+1}/{config.num_rep} done.")

    pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd = \
        compute_pooled_stats(pooled_raw, config.scalings, config.ratios, config.alpha_test)

    return results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd


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


def plot_norm_mmd(results_norm_mmd, pooled_norm_mmd, config, save_dir):
    rs = np.array(config.ratios)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        means_n = np.array([np.mean(results_norm_mmd[scal][r]) for r in rs])
        stds_n  = np.array([np.std(results_norm_mmd[scal][r]) for r in rs])
        ax.plot(rs, means_n, label=f"Scale: {scal}", color=color, marker='o')
        ax.fill_between(rs, means_n - stds_n, means_n + stds_n, color=color, alpha=0.2)
    ax.axvline(x=1.0, color='grey', linestyle=':', alpha=0.5, label=r'$\lambda_1/\lambda_0=1$')
    ax.set_xlabel(r"$\lambda_1 / \lambda_0$", fontsize=12)
    ax.set_ylabel(r"Normalized MMD ($\hat{MMD}^2 / \sigma_{H0}$)", fontsize=12)
    ax.set_title(f"Per-Rep Mean Normalized MMD ({config.num_rep} reps)", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "poisson_comparison_norm_mmd_per_rep.svg"), format="svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        pt_n = np.array([pooled_norm_mmd[scal][r] for r in rs])
        ax.plot(rs, pt_n, label=f"Scale: {scal}", color=color, marker='s')
    ax.axvline(x=1.0, color='grey', linestyle=':', alpha=0.5, label=r'$\lambda_1/\lambda_0=1$')
    ax.set_xlabel(r"$\lambda_1 / \lambda_0$", fontsize=12)
    ax.set_ylabel(r"Pooled Normalized MMD ($\hat{MMD}^2 / \sigma_{H0}$)", fontsize=12)
    ax.set_title(f"Pooled Normalized MMD vs Ratio", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "poisson_comparison_norm_mmd_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved normalized MMD plots to {save_dir}/")


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

    results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd = execute_sweep(config)

    plot_per_rep(results_t1e, results_t2e, config, kernel_dir)
    plot_pooled(pooled_t1e, pooled_t2e, config, kernel_dir)
    plot_pvalues(results_pval, pooled_pval, config, kernel_dir)
    plot_norm_mmd(results_norm_mmd, pooled_norm_mmd, config, kernel_dir)

    with open(os.path.join(kernel_dir, "metadata.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    logging.info("Poisson comparison experiment finished.")


if __name__ == "__main__":
    main()
