"""
Improved Hawkes Process Analysis - Signature MMD
- Uses configuration classes for better organization
- Plots Type 2 error vs parameter (alpha) for multiple scale values
- Precomputes Gram matrices for massive speedup (~100x)
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
    sim_hawkes_exp, process_paths_pair_to_tensor,
    precompute_gram_chunked, compute_errors_from_gram,
    init_results_dicts, accumulate_results, compute_pooled_stats,
    make_sig_kernel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    data_dir: str = "eq_hawkes_fixed_mean_improved"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Hawkes parameters
    target_mean: float = 100.0
    fixed_beta: float = 10.0
    alpha0: float = 5.0
    T: float = 10.0
    burn_in: float = 10.0
    grid_points: int = 300

    # Sweep configurations
    alphas_h1: List[float] = field(default_factory=lambda: np.linspace(2, 8, 7).tolist())
    scalings: List[float] = field(default_factory=lambda: [1.0])

    # Execution parameters
    n_atoms_delta: int = 1000
    n_paths: int = 256
    n_bank: int = 1024
    num_rep: int = 10
    alpha_test: float = 0.05

    # Kernel choice
    kernel_type: str = "rbf"
    rbf_sigma: float = 1.0

    def get_mu(self, alpha: float) -> float:
        return self.target_mean - self.fixed_beta * alpha

    @property
    def mu0(self) -> float:
        return self.get_mu(self.alpha0)

    def make_kernel(self, scaling: float = 1.0):
        return make_sig_kernel(self.kernel_type, self.rbf_sigma, scaling)


# ---------------------------------------------------------------------------
# Path loading
# ---------------------------------------------------------------------------
def load_hawkes_paths(config, num_sim, alpha1):
    mu1 = config.get_mu(alpha1)
    h0_bank = sim_hawkes_exp(config.mu0, config.alpha0, config.fixed_beta,
                             num_sim, config.grid_points, config.T, config.burn_in)
    h1_bank = sim_hawkes_exp(mu1, alpha1, config.fixed_beta,
                             num_sim, config.grid_points, config.T, config.burn_in)
    return process_paths_pair_to_tensor(h0_bank, h1_bank, config, num_sim)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def execute_sweep_multi_scaling(config):
    results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_raw = \
        init_results_dicts(config.scalings, config.alphas_h1)

    for rep in tqdm(range(config.num_rep), desc="Repetitions"):
        for alpha in tqdm(config.alphas_h1, desc=f"  Rep {rep+1} — alphas", leave=False):
            h0, h1 = load_hawkes_paths(config, num_sim=config.n_bank, alpha1=alpha)

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
                                   scal, alpha, t1e, t2e, mean_pval, raw)

        logging.info(f"Rep {rep+1}/{config.num_rep} done.")

    pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd = \
        compute_pooled_stats(pooled_raw, config.scalings, config.alphas_h1, config.alpha_test)

    total_paths = config.num_rep * config.n_bank
    total_atoms = config.num_rep * config.n_atoms_delta
    logging.info(f"Pooled test: {total_paths} total independent paths, {total_atoms} MMD values per (alpha, scaling)")

    return results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sweep_multi_scaling(results_t1e, results_t2e, config, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    alphas = np.array(config.alphas_h1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))

    for scal, color in zip(config.scalings, colors):
        means_t1 = np.array([np.mean(results_t1e[scal][a]) for a in alphas])
        stds_t1  = np.array([np.std(results_t1e[scal][a]) for a in alphas])
        axes[0].plot(alphas, means_t1, label=f"Scale: {scal}", color=color, marker='o')
        axes[0].fill_between(alphas, means_t1 - stds_t1, means_t1 + stds_t1, color=color, alpha=0.2)

        means_t2 = np.array([np.mean(results_t2e[scal][a]) for a in alphas])
        stds_t2  = np.array([np.std(results_t2e[scal][a]) for a in alphas])
        axes[1].plot(alphas, means_t2, label=f"Scale: {scal}", color=color, marker='o')
        axes[1].fill_between(alphas, means_t2 - stds_t2, means_t2 + stds_t2, color=color, alpha=0.2)

    axes[0].axhline(y=5.0, color='r', linestyle='--', label='5% Target')
    axes[0].axvline(x=config.alpha0, color='grey', linestyle='--', label=f'alpha_1={config.alpha0} (H1 = H0)')
    axes[0].set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    axes[0].set_ylabel("P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Type 1 Error vs Alpha ({config.num_rep} reps, {config.n_bank} paths/rep)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].axvline(x=config.alpha0, color='grey', linestyle='--', label=f'alpha_1={config.alpha0} (H1 = H0)')
    axes[1].set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    axes[1].set_ylabel("P[Type 2 Error] (%)", fontsize=12)
    axes[1].set_title(f"Type 2 Error vs Alpha ({config.num_rep} reps, {config.n_bank} paths/rep)", fontsize=13)
    axes[1].legend(title="Scaling", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "errors_vs_alpha_multi_scaling.svg"), format="svg")
    plt.close()
    logging.info(f"Saved sweep plot to {save_dir}/")


def plot_sweep_pooled(pooled_t1e, pooled_t2e, config, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    alphas = np.array(config.alphas_h1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))
    total_paths = config.num_rep * config.n_bank
    total_atoms = config.num_rep * config.n_atoms_delta

    for scal, color in zip(config.scalings, colors):
        pt1 = np.array([pooled_t1e[scal][a] for a in alphas])
        axes[0].plot(alphas, pt1, label=f"Scale: {scal}", color=color, marker='s')
        pt2 = np.array([pooled_t2e[scal][a] for a in alphas])
        axes[1].plot(alphas, pt2, label=f"Scale: {scal}", color=color, marker='s')

    axes[0].axhline(y=5.0, color='r', linestyle='--', label='5% Target')
    axes[0].axvline(x=config.alpha0, color='grey', linestyle='--', label=f'alpha_1={config.alpha0} (H1 = H0)')
    axes[0].set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    axes[0].set_ylabel("Pooled P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Pooled Global: Type 1 Error ({total_paths} paths, {total_atoms} MMD samples)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].axvline(x=config.alpha0, color='grey', linestyle='--', label=f'alpha_1={config.alpha0} (H1 = H0)')
    axes[1].set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    axes[1].set_ylabel("Pooled P[Type 2 Error] (%)", fontsize=12)
    axes[1].set_title(f"Pooled Global: Type 2 Error ({total_paths} paths, {total_atoms} MMD samples)", fontsize=13)
    axes[1].legend(title="Scaling", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "errors_vs_alpha_multi_scaling_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved pooled plot to {save_dir}/")


def plot_sweep_pvalues(results_pval, pooled_pval, config, save_dir):
    alphas = np.array(config.alphas_h1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        means_p = np.array([np.mean(results_pval[scal][a]) for a in alphas])
        stds_p  = np.array([np.std(results_pval[scal][a]) for a in alphas])
        ax.plot(alphas, means_p, label=f"Scale: {scal}", color=color, marker='o')
        ax.fill_between(alphas, means_p - stds_p, means_p + stds_p, color=color, alpha=0.2)
    ax.axvline(x=config.alpha0, color='grey', linestyle='--', label=f'alpha_1={config.alpha0} (H1 = H0)')
    ax.set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    ax.set_ylabel("Empirical P-value", fontsize=12)
    ax.set_title(f"Per-Rep Mean P-value vs Alpha ({config.num_rep} reps)", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "empirical_pvalues_per_rep.svg"), format="svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        pt_p = np.array([pooled_pval[scal][a] for a in alphas])
        ax.plot(alphas, pt_p, label=f"Scale: {scal}", color=color, marker='s')
    ax.axvline(x=config.alpha0, color='grey', linestyle='--', label=f'alpha_1={config.alpha0} (H1 = H0)')
    ax.set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    ax.set_ylabel("Pooled Empirical P-value", fontsize=12)
    ax.set_title(f"Pooled P-value vs Alpha", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "empirical_pvalues_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved separated p-values plots to {save_dir}/")


def plot_sweep_norm_mmd(results_norm_mmd, pooled_norm_mmd, config, save_dir):
    alphas = np.array(config.alphas_h1)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(config.scalings)))

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        means_n = np.array([np.mean(results_norm_mmd[scal][a]) for a in alphas])
        stds_n  = np.array([np.std(results_norm_mmd[scal][a]) for a in alphas])
        ax.plot(alphas, means_n, label=f"Scale: {scal}", color=color, marker='o')
        ax.fill_between(alphas, means_n - stds_n, means_n + stds_n, color=color, alpha=0.2)
    ax.axvline(x=config.alpha0, color='grey', linestyle='--', label=f'alpha_1={config.alpha0} (H1 = H0)')
    ax.set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    ax.set_ylabel(r"Normalized MMD ($\hat{MMD}^2 / \sigma_{H0}$)", fontsize=12)
    ax.set_title(f"Per-Rep Mean Normalized MMD ({config.num_rep} reps)", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "empirical_norm_mmd_per_rep.svg"), format="svg")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for scal, color in zip(config.scalings, colors):
        pt_n = np.array([pooled_norm_mmd[scal][a] for a in alphas])
        ax.plot(alphas, pt_n, label=f"Scale: {scal}", color=color, marker='s')
    ax.axvline(x=config.alpha0, color='grey', linestyle='--', label=f'alpha_1={config.alpha0} (H1 = H0)')
    ax.set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    ax.set_ylabel(r"Pooled Normalized MMD ($\hat{MMD}^2 / \sigma_{H0}$)", fontsize=12)
    ax.set_title(f"Pooled Normalized MMD vs Alpha", fontsize=13)
    ax.legend(title="Scaling", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "empirical_norm_mmd_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved normalized MMD plots to {save_dir}/")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    config = ExperimentConfig()
    os.makedirs(config.data_dir, exist_ok=True)

    logging.info(f"Starting Fixed Mean Evaluation. Target Mean = {config.target_mean}")
    logging.info(f"H0 setup: alpha0 = {config.alpha0}, mu0 = {config.mu0}")
    logging.info(f"Kernel: {config.kernel_type}" + (f" (sigma={config.rbf_sigma})" if config.kernel_type == "rbf" else ""))

    kernel_dir = os.path.join(config.data_dir, config.kernel_type)
    os.makedirs(kernel_dir, exist_ok=True)

    logging.info("Executing sweep over parameters for multiple scalings...")
    results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd = execute_sweep_multi_scaling(config)

    plot_sweep_multi_scaling(results_t1e, results_t2e, config, kernel_dir)
    plot_sweep_pooled(pooled_t1e, pooled_t2e, config, kernel_dir)
    plot_sweep_pvalues(results_pval, pooled_pval, config, kernel_dir)
    plot_sweep_norm_mmd(results_norm_mmd, pooled_norm_mmd, config, kernel_dir)

    with open(os.path.join(kernel_dir, "metadata.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    logging.info("Experiment finished completely.")

if __name__ == "__main__":
    main()
