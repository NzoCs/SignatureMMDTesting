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

from experiment_utils import (
    powerlaw_kernel, sim_hawkes_exp, sim_hawkes_general,
    process_paths_pair_to_tensor,
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
    branching_ratio: float = 0.5
    beta: float = 10.0
    T: float = 10.0
    burn_in: float = 10.0
    grid_points: int = 200

    # Sweep over power-law exponent p
    p_values: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0, 5.0, 8.0])
    scalings: List[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])

    # Execution
    n_atoms_delta: int = 1000
    n_paths: int = 256
    n_bank: int = 1024
    num_rep: int = 10
    alpha_test: float = 0.05

    # Kernel choice
    kernel_type: str = "rbf"
    rbf_sigma: float = 1.0

    @property
    def mu(self) -> float:
        return self.target_mean * (1 - self.branching_ratio)

    @property
    def alpha_exp(self) -> float:
        return self.branching_ratio * self.beta

    def alpha_poly(self, p: float) -> float:
        return self.branching_ratio * self.beta * (p - 1)

    def make_kernel(self, scaling: float = 1.0):
        return make_sig_kernel(self.kernel_type, self.rbf_sigma, scaling)


# ---------------------------------------------------------------------------
# Path loading
# ---------------------------------------------------------------------------
def load_paths(config, num_sim, p_value):
    """Generate H0 (exponential) and H1 (power-law) paths."""
    h0_bank = sim_hawkes_exp(
        config.mu, config.alpha_exp, config.beta, num_sim, config.grid_points, config.T, config.burn_in,
        desc=f"H0 exp(α={config.alpha_exp:.1f})"
    )

    a_poly = config.alpha_poly(p_value)
    poly_kern = powerlaw_kernel(a_poly, config.beta, p_value)
    h1_bank = sim_hawkes_general(
        config.mu, poly_kern, num_sim, config.grid_points, config.T, config.burn_in,
        desc=f"H1 poly(p={p_value:.1f})"
    )

    return process_paths_pair_to_tensor(h0_bank, h1_bank, config, num_sim)


# ---------------------------------------------------------------------------
# Sweep & Plotting
# ---------------------------------------------------------------------------
def execute_sweep(config):
    results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_raw = \
        init_results_dicts(config.scalings, config.p_values)

    for rep in tqdm(range(config.num_rep), desc="Repetitions"):
        for p_val in tqdm(config.p_values, desc=f"  Rep {rep+1} — p values", leave=False):
            h0, h1 = load_paths(config, num_sim=config.n_bank, p_value=p_val)

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
                                   scal, p_val, t1e, t2e, mean_pval, raw)

        logging.info(f"Rep {rep+1}/{config.num_rep} done.")

    pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd = \
        compute_pooled_stats(pooled_raw, config.scalings, config.p_values, config.alpha_test)

    return results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd


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


def plot_pvalues(results_pval, pooled_pval, config, save_dir):
    ps = np.array(config.p_values)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        means_p = np.array([np.mean(results_pval[scal][p]) for p in ps])
        stds_p  = np.array([np.std(results_pval[scal][p]) for p in ps])
        ax.plot(ps, means_p, label=f"Scale: {scal}", color=color, marker='o')
        ax.fill_between(ps, means_p - stds_p, means_p + stds_p, color=color, alpha=0.2)
    ax.set_xlabel("Power-law exponent p", fontsize=12)
    ax.set_ylabel("Empirical P-value", fontsize=12)
    ax.set_title(f"Per-Rep Mean P-value ({config.num_rep} reps)", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hawkes_kernel_pvalues_per_rep.svg"), format="svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        pt_p = np.array([pooled_pval[scal][p] for p in ps])
        ax.plot(ps, pt_p, label=f"Scale: {scal}", color=color, marker='s')
    ax.set_xlabel("Power-law exponent p", fontsize=12)
    ax.set_ylabel("Pooled Empirical P-value", fontsize=12)
    ax.set_title(f"Pooled P-value vs Power-law exp", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hawkes_kernel_pvalues_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved separated p-values plots to {save_dir}/")


def plot_norm_mmd(results_norm_mmd, pooled_norm_mmd, config, save_dir):
    ps = np.array(config.p_values)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        means_n = np.array([np.mean(results_norm_mmd[scal][p]) for p in ps])
        stds_n  = np.array([np.std(results_norm_mmd[scal][p]) for p in ps])
        ax.plot(ps, means_n, label=f"Scale: {scal}", color=color, marker='o')
        ax.fill_between(ps, means_n - stds_n, means_n + stds_n, color=color, alpha=0.2)
    ax.set_xlabel("Power-law exponent p", fontsize=12)
    ax.set_ylabel(r"Normalized MMD ($\hat{MMD}^2 / \sigma_{H0}$)", fontsize=12)
    ax.set_title(f"Per-Rep Mean Normalized MMD ({config.num_rep} reps)", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hawkes_kernel_norm_mmd_per_rep.svg"), format="svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        pt_n = np.array([pooled_norm_mmd[scal][p] for p in ps])
        ax.plot(ps, pt_n, label=f"Scale: {scal}", color=color, marker='s')
    ax.set_xlabel("Power-law exponent p", fontsize=12)
    ax.set_ylabel(r"Pooled Normalized MMD ($\hat{MMD}^2 / \sigma_{H0}$)", fontsize=12)
    ax.set_title(f"Pooled Normalized MMD vs Power-law exp", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hawkes_kernel_norm_mmd_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved normalized MMD plots to {save_dir}/")


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

    results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd = execute_sweep(config)

    plot_per_rep(results_t1e, results_t2e, config, kernel_dir)
    plot_pooled(pooled_t1e, pooled_t2e, config, kernel_dir)
    plot_pvalues(results_pval, pooled_pval, config, kernel_dir)
    plot_norm_mmd(results_norm_mmd, pooled_norm_mmd, config, kernel_dir)

    with open(os.path.join(kernel_dir, "metadata.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    logging.info("Kernel comparison experiment finished.")


if __name__ == "__main__":
    main()
