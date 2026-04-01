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

from src.mmd.mmd import SigKernel, LinearKernel, RBFKernel

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
    scalings: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 3.0, 4.0])
    
    # Execution parameters
    n_atoms_delta: int = 1000   # cheap with Gram precomputation, can be large
    n_paths: int = 256         # batch size for each MMD sub-sample
    n_bank: int = 1024          # total paths generated per rep (more = more independent sub-samples)
    num_rep: int = 10          # repetitions with fresh paths for confidence intervals
    alpha_test: float = 0.05
    
    # Kernel choice
    kernel_type: str = "rbf"   # "linear" or "rbf"
    rbf_sigma: float = 1.0        # sigma for RBF kernel
    
    def get_mu(self, alpha: float) -> float:
        """Calculate mu to keep the implied mean constant."""
        return self.target_mean - self.fixed_beta * alpha

    @property
    def mu0(self) -> float:
        return self.get_mu(self.alpha0)

    def make_kernel(self, scaling: float = 1.0) -> SigKernel:
        """Create a SigKernel with the configured static kernel and given scaling."""
        if self.kernel_type == "rbf":
            static = RBFKernel(sigma=self.rbf_sigma, scaling=scaling)
        else:
            static = LinearKernel(scaling=scaling)
        return SigKernel(static_kernel=static, dyadic_order=0)

# ---------------------------------------------------------------------------
# Simulator
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

def sim_hawkes_model(mu, alpha, beta, num_sim, num_time_steps, T, burn_in=100.0):
    time_grid = np.linspace(0, T, num_time_steps)
    paths = np.zeros((num_time_steps, num_sim))

    for s in tqdm(range(num_sim), desc=f"Simulating Hawkes (mu={mu:.1f}, alpha={alpha:.1f})", leave=False):
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

def load_hawkes_paths(config: ExperimentConfig, num_sim: int, alpha1: float):
    mu1 = config.get_mu(alpha1)
    
    h0_bank = sim_hawkes_model(config.mu0, config.alpha0, config.fixed_beta, num_sim, config.grid_points, config.T, config.burn_in)
    h1_bank = sim_hawkes_model(mu1, alpha1, config.fixed_beta, num_sim, config.grid_points, config.T, config.burn_in)

    h0 = torch.transpose(torch.from_numpy(h0_bank), 0, 1).to(device=config.device, dtype=torch.float32)
    h1 = torch.transpose(torch.from_numpy(h1_bank), 0, 1).to(device=config.device, dtype=torch.float32)

    for i in range(num_sim):
        h0[i] -= h0[i, 0, :]
        h1[i] -= h1[i, 0, :]

    # Normalize
    count_std = h0[:, -1, 0].std().item()
    if count_std > 1e-8:
        h0[:, :, 0] /= count_std
        h1[:, :, 0] /= count_std
    if config.T > 0:
        h0[:, :, 1] /= config.T
        h1[:, :, 1] /= config.T

    return h0, h1

# ---------------------------------------------------------------------------
# Gram precomputation helpers
# ---------------------------------------------------------------------------
def precompute_gram_chunked(sig_kernel, X, Y, sym=False, chunk_size=64):
    """
    Precompute the full signature kernel Gram matrix in chunks.
    This is the ONLY expensive computation — done once per (alpha, scaling).
    """
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
    """
    Compute unbiased MMD^2 estimate directly from sub-matrices of a
    precomputed Gram matrix.  This is O(n^2) additions — essentially free.
    """
    nx = K_XX.shape[0]
    ny = K_YY.shape[0]
    xx = (K_XX.sum() - K_XX.diagonal().sum()) / (nx * (nx - 1))
    yy = (K_YY.sum() - K_YY.diagonal().sum()) / (ny * (ny - 1))
    xy = K_XY.mean()
    return (xx + yy - 2.0 * xy).item()


def compute_errors_from_gram(K_h0, K_h1, K_h0h1, n_atoms, batch_size, alpha_test):
    """
    Compute Type 1 and Type 2 error rates by sub-sampling indices
    from precomputed Gram matrices.  The sub-sampling is virtually free
    compared to kernel evaluation.
    
    K_h0:   (n0, n0) Gram matrix of H0 paths
    K_h1:   (n1, n1) Gram matrix of H1 paths
    K_h0h1: (n0, n1) cross Gram matrix
    """
    n0 = K_h0.shape[0]
    n1 = K_h1.shape[0]
    
    # Move to CPU for fast sub-sampling
    K_h0 = K_h0.cpu()
    K_h1 = K_h1.cpu()
    K_h0h1 = K_h0h1.cpu()
    
    h0_dists = np.empty(n_atoms)
    h1_dists = np.empty(n_atoms)
    h00_dists = np.empty(n_atoms)
    h01_dists = np.empty(n_atoms)
    
    for i in range(n_atoms):
        # --- Type 2 error test: H0 vs H1 ---
        ix1 = torch.randperm(n0)[:batch_size]
        ix2 = torch.randperm(n0)[:batch_size]
        iy  = torch.randperm(n1)[:batch_size]
        
        # Null: MMD(h0[ix1], h0[ix2])
        h0_dists[i] = mmd_ub_from_subgram(
            K_h0[ix1][:, ix1], K_h0[ix2][:, ix2], K_h0[ix1][:, ix2]
        )
        # Alt: MMD(h0[ix1], h1[iy])
        h1_dists[i] = mmd_ub_from_subgram(
            K_h0[ix1][:, ix1], K_h1[iy][:, iy], K_h0h1[ix1][:, iy]
        )
        
        # --- Type 1 error test: H0 vs H0 ---
        ix3 = torch.randperm(n0)[:batch_size]
        ix4 = torch.randperm(n0)[:batch_size]
        ix5 = torch.randperm(n0)[:batch_size]
        
        h00_dists[i] = mmd_ub_from_subgram(
            K_h0[ix3][:, ix3], K_h0[ix4][:, ix4], K_h0[ix3][:, ix4]
        )
        h01_dists[i] = mmd_ub_from_subgram(
            K_h0[ix3][:, ix3], K_h0[ix5][:, ix5], K_h0[ix3][:, ix5]
        )
    
    # Type 2 error
    crit = np.sort(h0_dists)[int(n_atoms * (1 - alpha_test))]
    t2e = 100.0 * np.mean(h1_dists <= crit)
    
    # Type 1 error
    crit2 = np.sort(h00_dists)[int(n_atoms * (1 - alpha_test))]
    t1e = 100.0 * np.mean(h01_dists <= crit2)
    
    # Return errors AND raw distributions for pooling
    raw = (h0_dists, h1_dists, h00_dists, h01_dists)
    return 100.0 - t1e, t2e, raw


# ---------------------------------------------------------------------------
# Analysis & Plotting
# ---------------------------------------------------------------------------
def execute_sweep_multi_scaling(config: ExperimentConfig):
    """
    For each rep: generate fresh independent paths, then for each (alpha, scaling):
      1. Precompute 3 Gram matrices (the only expensive step)
      2. Sub-sample indices n_atoms times to build MMD distributions (free)
      3. Compute error rates
    Multiple reps with fresh paths give proper confidence intervals.
    Also accumulates raw MMD distributions across all reps for a global pooled test.
    """
    results_t1e = {s: {a: [] for a in config.alphas_h1} for s in config.scalings}
    results_t2e = {s: {a: [] for a in config.alphas_h1} for s in config.scalings}
    
    # Accumulate raw MMD distributions across all reps for pooled test
    pooled_raw = {s: {a: {'h0': [], 'h1': [], 'h00': [], 'h01': []} 
                      for a in config.alphas_h1} for s in config.scalings}
    
    for rep in tqdm(range(config.num_rep), desc="Repetitions"):
        for alpha in tqdm(config.alphas_h1, desc=f"  Rep {rep+1} — alphas", leave=False):
            # Generate FRESH independent paths each rep
            h0, h1 = load_hawkes_paths(config, num_sim=config.n_bank, alpha1=alpha)
            
            for scal in config.scalings:
                # Use kernel scaling parameter via config
                scaled_kernel = config.make_kernel(scaling=scal)
                
                # Precompute Gram matrices ONCE per (rep, alpha, scaling)
                K_h0   = precompute_gram_chunked(scaled_kernel, h0, h0, sym=True)
                K_h1   = precompute_gram_chunked(scaled_kernel, h1, h1, sym=True)
                K_h0h1 = precompute_gram_chunked(scaled_kernel, h0, h1, sym=False)
                
                # Sub-sample from precomputed Grams — virtually free
                t1e, t2e, raw = compute_errors_from_gram(
                    K_h0, K_h1, K_h0h1,
                    n_atoms=config.n_atoms_delta,
                    batch_size=config.n_paths,
                    alpha_test=config.alpha_test
                )
                
                results_t1e[scal][alpha].append(t1e)
                results_t2e[scal][alpha].append(t2e)
                
                # Accumulate raw MMD values for pooled global test
                h0_d, h1_d, h00_d, h01_d = raw
                pooled_raw[scal][alpha]['h0'].append(h0_d)
                pooled_raw[scal][alpha]['h1'].append(h1_d)
                pooled_raw[scal][alpha]['h00'].append(h00_d)
                pooled_raw[scal][alpha]['h01'].append(h01_d)
                
        logging.info(f"Rep {rep+1}/{config.num_rep} done.")
    
    # Compute pooled errors from ALL accumulated MMD values
    pooled_t1e = {s: {} for s in config.scalings}
    pooled_t2e = {s: {} for s in config.scalings}
    
    for scal in config.scalings:
        for alpha in config.alphas_h1:
            all_h0  = np.concatenate(pooled_raw[scal][alpha]['h0'])
            all_h1  = np.concatenate(pooled_raw[scal][alpha]['h1'])
            all_h00 = np.concatenate(pooled_raw[scal][alpha]['h00'])
            all_h01 = np.concatenate(pooled_raw[scal][alpha]['h01'])
            
            n_total = len(all_h0)
            crit = np.sort(all_h0)[int(n_total * (1 - config.alpha_test))]
            t2e = 100.0 * np.mean(all_h1 <= crit)
            
            crit2 = np.sort(all_h00)[int(n_total * (1 - config.alpha_test))]
            t1e = 100.0 * np.mean(all_h01 <= crit2)
            
            pooled_t1e[scal][alpha] = 100.0 - t1e
            pooled_t2e[scal][alpha] = t2e
    
    total_paths = config.num_rep * config.n_bank
    total_atoms = config.num_rep * config.n_atoms_delta
    logging.info(f"Pooled test: {total_paths} total independent paths, {total_atoms} MMD values per (alpha, scaling)")
    
    return results_t1e, results_t2e, pooled_t1e, pooled_t2e


def plot_sweep_multi_scaling(results_t1e, results_t2e, config: ExperimentConfig, save_dir: str):
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
        
    axes[0].set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    axes[0].set_ylabel("P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Type 1 Error vs Alpha ({config.num_rep} reps, {config.n_bank} paths/rep)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    axes[1].set_ylabel("P[Type 2 Error] (%)", fontsize=12)
    axes[1].set_title(f"Type 2 Error vs Alpha ({config.num_rep} reps, {config.n_bank} paths/rep)", fontsize=13)
    axes[1].legend(title="Scaling", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "errors_vs_alpha_multi_scaling.svg"), format="svg")
    plt.close()
    logging.info(f"Saved sweep plot to {save_dir}/")


def plot_sweep_pooled(pooled_t1e, pooled_t2e, config: ExperimentConfig, save_dir: str):
    """Plot error rates from the pooled global test (all reps combined)."""
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
        
    axes[0].set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    axes[0].set_ylabel("Pooled P[Type 1 Error] (%)", fontsize=12)
    axes[0].set_title(f"Pooled Global: Type 1 Error ({total_paths} paths, {total_atoms} MMD samples)", fontsize=13)
    axes[0].legend(title="Scaling", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].set_xlabel(r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)", fontsize=12)
    axes[1].set_ylabel("Pooled P[Type 2 Error] (%)", fontsize=12)
    axes[1].set_title(f"Pooled Global: Type 2 Error ({total_paths} paths, {total_atoms} MMD samples)", fontsize=13)
    axes[1].legend(title="Scaling", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "errors_vs_alpha_multi_scaling_pooled.svg"), format="svg")
    plt.close()
    logging.info(f"Saved pooled plot to {save_dir}/")


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
    results_t1e, results_t2e, pooled_t1e, pooled_t2e = execute_sweep_multi_scaling(config)
    
    # Plot 1: per-rep mean ± std (confidence intervals)
    plot_sweep_multi_scaling(results_t1e, results_t2e, config, kernel_dir)
    
    # Plot 2: pooled global test (all reps combined)
    plot_sweep_pooled(pooled_t1e, pooled_t2e, config, kernel_dir)
    
    with open(os.path.join(kernel_dir, "metadata.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)
        
    logging.info("Experiment finished completely.")

if __name__ == "__main__":
    main()
