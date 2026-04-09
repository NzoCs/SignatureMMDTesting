"""
Shared utilities for Signature MMD experiments.

Contains:
- Hawkes process simulation (exponential and general kernels)
- Kernel functions (exponential, power-law)
- Path processing (numpy → tensor, centering, normalization)
- Gram matrix precomputation
- MMD computation from Gram sub-matrices
- Error rate computation (Type 1, Type 2, p-values, normalized MMD)
- Pooled statistics computation
"""

import logging
import numpy as np
import torch
from tqdm import tqdm

from src.mmd.mmd import SigKernel, LinearKernel, RBFKernel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------
def powerlaw_kernel(alpha, beta, p):
    def kernel(dt):
        return alpha * (1.0 + beta * dt) ** (-p)
    return kernel


# ---------------------------------------------------------------------------
# SigKernel factory
# ---------------------------------------------------------------------------
def make_sig_kernel(kernel_type, rbf_sigma=1.0, scaling=1.0):
    """Create a SigKernel with the given static kernel type and scaling."""
    if kernel_type == "rbf":
        static = RBFKernel(sigma=rbf_sigma, scaling=scaling)
    else:
        static = LinearKernel(scaling=scaling)
    return SigKernel(static_kernel=static, dyadic_order=0)


# ---------------------------------------------------------------------------
# Hawkes Simulator (Exponential kernel - exact)
# ---------------------------------------------------------------------------
class HawkesSimulator:
    """Simulate 1D Hawkes process with exponential kernel using thinning."""

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


def sim_hawkes_exp(mu, alpha, beta, num_sim, num_time_steps, T, burn_in=100.0, desc=""):
    """Simulate Hawkes process with exponential kernel (exact, fast)."""
    if not desc:
        desc = f"Simulating Hawkes (mu={mu:.1f}, alpha={alpha:.1f})"
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
# Ogata Thinning Simulator (General kernel)
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
    """Simulate Hawkes process with arbitrary kernel via Ogata thinning."""
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


# ---------------------------------------------------------------------------
# Path processing
# ---------------------------------------------------------------------------
def process_paths_to_tensor(h_bank, config, num_sim):
    """Convert path bank (numpy) to tensor, center starting points, normalize.

    Config must have .device and .T attributes.
    """
    h_tensor = torch.transpose(torch.from_numpy(h_bank), 0, 1).to(
        device=config.device, dtype=torch.float32
    )

    for i in range(num_sim):
        h_tensor[i] -= h_tensor[i, 0, :]

    count_std = h_tensor[:, -1, 0].std().item()
    if count_std > 1e-8:
        h_tensor[:, :, 0] /= count_std
    if config.T > 0:
        h_tensor[:, :, 1] /= config.T

    return h_tensor


def process_paths_pair_to_tensor(h0_bank, h1_bank, config, num_sim):
    """Convert H0 and H1 path banks to tensors, normalizing H1 by H0's std.

    Config must have .device and .T attributes.
    """
    h0 = torch.transpose(torch.from_numpy(h0_bank), 0, 1).to(
        device=config.device, dtype=torch.float32
    )
    h1 = torch.transpose(torch.from_numpy(h1_bank), 0, 1).to(
        device=config.device, dtype=torch.float32
    )

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
def precompute_gram_chunked(sig_kernel, X, Y, sym=False, chunk_size=64):
    """Precompute the full signature kernel Gram matrix in chunks."""
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
    """Compute unbiased MMD^2 estimate from sub-matrices of a precomputed Gram matrix."""
    nx = K_XX.shape[0]
    ny = K_YY.shape[0]
    xx = (K_XX.sum() - K_XX.diagonal().sum()) / (nx * (nx - 1))
    yy = (K_YY.sum() - K_YY.diagonal().sum()) / (ny * (ny - 1))
    xy = K_XY.mean()
    return (xx + yy - 2.0 * xy).item()


def compute_errors_from_gram(K_h0, K_h1, K_h0h1, n_atoms, batch_size, alpha_test):
    """Compute Type 1/2 error rates by sub-sampling from precomputed Gram matrices.

    Returns: (type1_error%, type2_error%, mean_pvalue, raw_distributions)
    """
    n0 = K_h0.shape[0]
    n1 = K_h1.shape[0]

    K_h0 = K_h0.cpu()
    K_h1 = K_h1.cpu()
    K_h0h1 = K_h0h1.cpu()

    h0_dists = np.empty(n_atoms)
    h1_dists = np.empty(n_atoms)
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

    crit = np.sort(h0_dists)[int(n_atoms * (1 - alpha_test))]
    t2e = 100.0 * np.mean(h1_dists <= crit)

    crit2 = np.sort(h00_dists)[int(n_atoms * (1 - alpha_test))]
    t1e = 100.0 * np.mean(h01_dists <= crit2)

    h0_sorted = np.sort(h0_dists)
    p_val_arr = (n_atoms - np.searchsorted(h0_sorted, h1_dists, side='left')) / n_atoms
    mean_pval = np.mean(p_val_arr)

    raw = (h0_dists, h1_dists, h00_dists, h01_dists)
    return 100.0 - t1e, t2e, mean_pval, raw


# ---------------------------------------------------------------------------
# Results aggregation helpers
# ---------------------------------------------------------------------------
def init_results_dicts(scalings, param_values):
    """Initialize per-rep results and pooled_raw dictionaries."""
    results_t1e = {s: {p: [] for p in param_values} for s in scalings}
    results_t2e = {s: {p: [] for p in param_values} for s in scalings}
    results_pval = {s: {p: [] for p in param_values} for s in scalings}
    results_norm_mmd = {s: {p: [] for p in param_values} for s in scalings}
    pooled_raw = {s: {p: {'h0': [], 'h1': [], 'h00': [], 'h01': []}
                      for p in param_values} for s in scalings}
    return results_t1e, results_t2e, results_pval, results_norm_mmd, pooled_raw


def accumulate_results(results_t1e, results_t2e, results_pval, results_norm_mmd,
                       pooled_raw, scal, param, t1e, t2e, mean_pval, raw):
    """Accumulate results from a single (scaling, param) computation."""
    results_t1e[scal][param].append(t1e)
    results_t2e[scal][param].append(t2e)
    results_pval[scal][param].append(mean_pval)

    h0_d, h1_d, h00_d, h01_d = raw
    if np.std(h0_d) > 1e-12:
        results_norm_mmd[scal][param].append(np.mean(h1_d) / np.std(h0_d))
    else:
        results_norm_mmd[scal][param].append(0.0)

    pooled_raw[scal][param]['h0'].append(h0_d)
    pooled_raw[scal][param]['h1'].append(h1_d)
    pooled_raw[scal][param]['h00'].append(h00_d)
    pooled_raw[scal][param]['h01'].append(h01_d)


def compute_pooled_stats(pooled_raw, scalings, param_values, alpha_test):
    """Compute pooled error rates from accumulated raw MMD distributions."""
    pooled_t1e = {s: {} for s in scalings}
    pooled_t2e = {s: {} for s in scalings}
    pooled_pval = {s: {} for s in scalings}
    pooled_norm_mmd = {s: {} for s in scalings}

    for scal in scalings:
        for param in param_values:
            a0  = np.concatenate(pooled_raw[scal][param]['h0'])
            a1  = np.concatenate(pooled_raw[scal][param]['h1'])
            a00 = np.concatenate(pooled_raw[scal][param]['h00'])
            a01 = np.concatenate(pooled_raw[scal][param]['h01'])
            n = len(a0)
            sorted_a0 = np.sort(a0)
            c1 = sorted_a0[int(n * (1 - alpha_test))]
            c2 = np.sort(a00)[int(n * (1 - alpha_test))]
            pooled_t2e[scal][param] = 100.0 * np.mean(a1 <= c1)
            pooled_t1e[scal][param] = 100.0 - 100.0 * np.mean(a01 <= c2)

            p_vals_pooled = (n - np.searchsorted(sorted_a0, a1, side='left')) / n
            pooled_pval[scal][param] = np.mean(p_vals_pooled)

            if np.std(a0) > 1e-12:
                pooled_norm_mmd[scal][param] = np.mean(a1) / np.std(a0)
            else:
                pooled_norm_mmd[scal][param] = 0.0

    return pooled_t1e, pooled_t2e, pooled_pval, pooled_norm_mmd
