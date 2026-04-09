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

from experiment_utils import (
    powerlaw_kernel, sim_hawkes_exp, sim_hawkes_general,
    process_paths_to_tensor,
    precompute_gram_chunked, compute_errors_from_gram,
    init_results_dicts, accumulate_results, compute_pooled_stats,
    make_sig_kernel,
)

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
    scalings: List[float] = field(default_factory=lambda: [1.0])

    # Execution
    n_atoms_delta: int = 1000
    n_paths: int = 128
    n_bank: int = 1024
    num_rep: int = 10
    alpha_test: float = 0.05

    # Kernel choice
    kernel_type: str = "rbf"
    rbf_sigma: float = 1.0

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

    def make_kernel(self, scaling: float = 1.0):
        return make_sig_kernel(self.kernel_type, self.rbf_sigma, scaling)


# ---------------------------------------------------------------------------
# Sweep & Plotting
# ---------------------------------------------------------------------------
def execute_sweep(config):
    results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_raw = \
        init_results_dicts(config.scalings, config.alphas_h1)

    poly_kern = powerlaw_kernel(config.alpha0_poly, config.beta0, config.p0)

    for rep in tqdm(range(config.num_rep), desc="Repetitions"):
        # Simulate H0 ONCE per rep
        h0_bank_raw = sim_hawkes_general(
            config.mu0, poly_kern, config.n_bank, config.grid_points, config.T, config.burn_in,
            desc=f"Rep {rep+1} - H0 Power-law (p={config.p0})"
        )
        h0 = process_paths_to_tensor(h0_bank_raw, config, config.n_bank)

        for alpha1 in tqdm(config.alphas_h1, desc=f"  Rep {rep+1} — H1 alphas", leave=False):
            beta1 = config.get_beta1(alpha1)
            h1_bank_raw = sim_hawkes_exp(
                config.mu1, alpha1, beta1, config.n_bank, config.grid_points, config.T, config.burn_in,
                desc=f"H1 Exp (α={alpha1:.1f}, β={beta1:.1f})"
            )
            h1 = process_paths_to_tensor(h1_bank_raw, config, config.n_bank)

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
                                   scal, alpha1, t1e, t2e, mean_pval, raw)

        logging.info(f"Rep {rep+1}/{config.num_rep} done.")

    pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd = \
        compute_pooled_stats(pooled_raw, config.scalings, config.alphas_h1, config.alpha_test)

    return results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd


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
    axes[0].set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    axes[0].set_ylabel("P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Type 1 Error — PL vs Exp ({config.num_rep} reps)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
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
    axes[0].set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    axes[0].set_ylabel("Pooled P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Pooled: Type 1 Error ({total_paths} paths, {total_atoms} MMD)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
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

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        means_p = np.array([np.mean(results_pval[scal][x]) for x in xs])
        stds_p  = np.array([np.std(results_pval[scal][x]) for x in xs])
        ax.plot(xs, means_p, label=f"Scale: {scal}", color=color, marker='o')
        ax.fill_between(xs, means_p - stds_p, means_p + stds_p, color=color, alpha=0.2)
    ax.set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    ax.set_ylabel("Empirical P-value", fontsize=12)
    ax.set_title(f"Per-Rep Mean P-value ({config.num_rep} reps)", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_comp_opt_pvalues_per_rep.svg"), format="svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        pt_p = np.array([pooled_pval[scal][x] for x in xs])
        ax.plot(xs, pt_p, label=f"Scale: {scal}", color=color, marker='s')
    ax.set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    ax.set_ylabel("Pooled Empirical P-value", fontsize=12)
    ax.set_title(f"Pooled P-value vs Alpha", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_comp_opt_pvalues_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved separated p-values plots to {save_dir}/")


def plot_norm_mmd(results_norm_mmd, pooled_norm_mmd, config, save_dir):
    xs = np.array(config.alphas_h1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        means_n = np.array([np.mean(results_norm_mmd[scal][x]) for x in xs])
        stds_n  = np.array([np.std(results_norm_mmd[scal][x]) for x in xs])
        ax.plot(xs, means_n, label=f"Scale: {scal}", color=color, marker='o')
        ax.fill_between(xs, means_n - stds_n, means_n + stds_n, color=color, alpha=0.2)
    ax.set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    ax.set_ylabel(r"Normalized MMD ($\hat{MMD}^2 / \sigma_{H0}$)", fontsize=12)
    ax.set_title(f"Per-Rep Mean Normalized MMD ({config.num_rep} reps)", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_comp_opt_norm_mmd_per_rep.svg"), format="svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        pt_n = np.array([pooled_norm_mmd[scal][x] for x in xs])
        ax.plot(xs, pt_n, label=f"Scale: {scal}", color=color, marker='s')
    ax.set_xlabel(r"H1 base intensity $\alpha_1$", fontsize=12)
    ax.set_ylabel(r"Pooled Normalized MMD ($\hat{MMD}^2 / \sigma_{H0}$)", fontsize=12)
    ax.set_title(f"Pooled Normalized MMD vs Alpha", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kernel_comp_opt_norm_mmd_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved normalized MMD plots to {save_dir}/")


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

    results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd = execute_sweep(config)

    plot_per_rep(results_t1e, results_t2e, config, kernel_dir)
    plot_pooled(pooled_t1e, pooled_t2e, config, kernel_dir)
    plot_pvalues(results_pval, pooled_pval, config, kernel_dir)
    plot_norm_mmd(results_norm_mmd, pooled_norm_mmd, config, kernel_dir)

    with open(os.path.join(kernel_dir, "metadata_opt.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    logging.info("Optimized Kernel comparison experiment finished.")


if __name__ == "__main__":
    main()
