"""
Hawkes Process Analysis - Signature MMD Two-Sample Statistical Tests
Andrew Alden, Blanka Horvath, Zacharia Issa

Two experiment configurations, each run with Linear kernel AND RBF kernel:
  A. strong_T1  : high-rate params (mu=20, alpha~0.7, beta=10), T=1,
                  paths normalised to O(1)
  B. normal_T20 : standard params  (mu=2,  alpha~0.3, beta=5),  T=20,
                  paths normalised to O(1)

For each config × kernel combination, 4 scenarios + 2 sensitivity sweeps:
  Scenario 1 — same mu, different alpha
  Scenario 2 — different mu, same alpha
  Scenario 3 — different mu AND different alpha
  Scenario 4 — same mu & alpha, different beta

  Sensitivity Δmu  — errors vs mu difference (alpha, beta fixed)
  Sensitivity Δbeta — errors vs beta difference (mu, alpha fixed)

All MMD estimates use the unbiased estimator.
    A summary txt file is written to eq_hawkes/analysis_summary.txt.
"""

import datetime
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless cluster execution
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from src.utils.helper_functions.plot_helper_functions import make_grid, golden_dimensions
from src.utils.plotting_functions import (
    plot_dist,
    plot_level_contributions,
)
from src.mmd.distribution_functions import (
    return_mmd_distributions,
    expected_type2_error,
    get_level_values,
    generate_error_probs_linear_kernel,
    get_type1_type2_errors,
)
from src.mmd.mmd import SigKernel, LinearKernel, RBFKernel

# ---------------------------------------------------------------------------
# Device & kernels
# ---------------------------------------------------------------------------
# Output directory for analysis results (change here to adjust output folder)
DATA_DIR = "eq_hawkes"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

dyadic_order = 0
sig_kernel_linear = SigKernel(static_kernel=LinearKernel(), dyadic_order=dyadic_order)
sig_kernel_rbf    = SigKernel(static_kernel=RBFKernel(sigma=1.0),  dyadic_order=dyadic_order)

KERNELS = [("linear", sig_kernel_linear), ("rbf", sig_kernel_rbf)]

# Module-level alias kept for any internal helper that may reference it
signature_kernel = sig_kernel_rbf

# ---------------------------------------------------------------------------
# Global NaN / diagnostic log
# ---------------------------------------------------------------------------
nan_log = []  # list of dicts: {tag, removed, total}

# ---------------------------------------------------------------------------
# Hawkes process simulator (Ogata thinning algorithm)
# ---------------------------------------------------------------------------

class HawkesSimulator:
    """
    Multivariate Hawkes process simulator using Ogata's thinning algorithm.
    Intensity: λ_i(t) = μ_i + Σ_j Σ_{t_k^j < t} α_{ij} * exp(-β_{ij} * (t - t_k^j))
    """

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
        """
        Simulate a multivariate Hawkes process.

        Returns:
            (times, marks): event times and corresponding dimension indices.
        """
        dim = self.dim_process
        times = []
        marks = []

        t = self.start_time
        lambda_trg = np.ones((dim, dim))

        while t < self.end_time:
            lambda_total = np.array(
                [self.mu[i] + np.sum(lambda_trg[i]) for i in range(dim)]
            )
            lambda_sum = np.sum(lambda_total)

            dt = (
                np.random.exponential(scale=1.0 / lambda_sum)
                if lambda_sum > 0
                else float("inf")
            )
            t = t + dt

            if t >= self.end_time:
                break

            lambda_trg *= np.exp(-self.beta * dt)

            lambda_next = np.array(
                [self.mu[i] + np.sum(lambda_trg[i]) for i in range(dim)]
            )
            lambda_next_sum = np.sum(lambda_next)

            if np.random.rand() < lambda_next_sum / lambda_sum:
                event_dim = np.random.choice(dim, p=lambda_total / lambda_sum)
                times.append(t)
                marks.append(event_dim)
                lambda_trg[:, event_dim] += self.alpha[:, event_dim]

        return np.array(times), np.array(marks)


def sim_hawkes_model(mu, alpha, beta, num_sim, num_time_steps, T):
    """
    Simulate Hawkes process count paths on [0, T] using Ogata's thinning algorithm.
    Intensity: λ(t) = μ + α * Σ_{t_i < t} exp(-β * (t - t_i))

    Returns array of shape (num_time_steps, num_sim, 2) — (count, time).
    """
    time_grid = np.linspace(0, T, num_time_steps)
    paths = np.zeros((num_time_steps, num_sim))

    for s in range(num_sim):
        simulator = HawkesSimulator(
            mu=[mu], alpha=[[alpha]], beta=[[beta]],
            dim_process=1, start_time=0.0, end_time=T,
        )
        event_times, _ = simulator.simulate()
        if len(event_times) > 0:
            paths[:, s] = np.searchsorted(event_times, time_grid, side="right")

    return np.concatenate(
        (paths[:, :, None],
         np.repeat(time_grid[:, None, None], repeats=num_sim, axis=1)),
        axis=2,
    )


def load_hawkes_paths(mu0, alpha0, mu1, alpha1, beta0, path_bank_size, grid_points, T,
                       beta1=None, normalize=True):
    """Simulate and return centred (and optionally normalised) torch path banks.

    If normalize=True:
      - count channel divided by empirical std of final count (H0 bank)
      - time  channel divided by T  (so time lives in [0, 1])
    This ensures both channels are O(1) regardless of T or event-rate.
    beta1 defaults to beta0 when omitted.
    """
    if beta1 is None:
        beta1 = beta0
    h0_bank = sim_hawkes_model(mu0, alpha0, beta0, path_bank_size, grid_points, T)
    h1_bank = sim_hawkes_model(mu1, alpha1, beta1, path_bank_size, grid_points, T)

    h0 = torch.transpose(torch.from_numpy(h0_bank), 0, 1).to(device=device, dtype=torch.float32)
    h1 = torch.transpose(torch.from_numpy(h1_bank), 0, 1).to(device=device, dtype=torch.float32)

    for i in range(path_bank_size):
        h0[i] = h0[i] - h0[i, 0, :]
        h1[i] = h1[i] - h1[i, 0, :]

    if normalize:
        # Normalise count channel by empirical std of final counts (H0 bank)
        count_std = h0[:, -1, 0].std().item()
        if count_std > 1e-8:
            h0[:, :, 0] = h0[:, :, 0] / count_std
            h1[:, :, 0] = h1[:, :, 0] / count_std
        # Normalise time channel to [0, 1]
        if T > 0:
            h0[:, :, 1] = h0[:, :, 1] / T
            h1[:, :, 1] = h1[:, :, 1] / T

    return h0, h1


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def filter_nan(dists, tag=""):
    """Remove NaN / inf values and log the event to nan_log."""
    clean = [d for d in dists if np.isfinite(d)]
    n_removed = len(dists) - len(clean)
    nan_log.append({"tag": tag, "removed": n_removed, "total": len(dists)})
    if n_removed > 0:
        print(f"  Warning: removed {n_removed}/{len(dists)} non-finite MMD values. [{tag}]")
    return clean


def plot_type2_error_n(type2_list, scalings, n_paths_list, num_sim,
                        title="", filename=None, colors=None):
    """plot_type2_error with configurable num_sim (not hardcoded to 100)."""
    if colors is None:
        colors = ["magenta", "green", "darkorange", "blue"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, n_paths in enumerate(n_paths_list):
        t2e = [type2_list[j][n_paths] for j in range(num_sim)]
        t2e_mean = np.mean(np.asarray(t2e), axis=0)
        t2e_std  = np.std(np.asarray(t2e), axis=0)
        ax.plot(scalings, t2e_mean, alpha=1, label=f"{n_paths}", color=colors[i])
        ax.fill_between(scalings, t2e_mean - t2e_std, t2e_mean + t2e_std,
                         color=colors[i], alpha=0.3, edgecolor="none")
    ax.set_ylabel("P[Type 2 Error] (%)", fontsize=13)
    ax.set_xlabel("Scaling", fontsize=13)
    ax.legend(loc="best", fontsize=13)
    ax.grid(True, color="black", alpha=0.2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.title(title, fontsize=13)
    if filename:
        plt.savefig(filename, bbox_inches="tight", format="svg")
    plt.close()


def plot_type1_error_n(type1_list, scalings, n_paths_list, num_sim,
                        title="", filename=None, colors=None):
    """plot_type1_error with configurable num_sim and n_paths_list length."""
    if colors is None:
        colors = ["magenta", "green", "darkorange", "blue"]
    from matplotlib.ticker import FormatStrFormatter
    n = len(n_paths_list)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for i, n_paths in enumerate(n_paths_list):
        ax = axes_flat[i]
        t1e = [100 - np.asarray(type1_list[j][n_paths]) for j in range(num_sim)]
        bp = ax.boxplot(np.asarray(t1e)[:, 1::2], patch_artist=True,
                        labels=np.round(scalings[1::2], 2))
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.3)
        for median in bp["medians"]:
            median.set(color=colors[i % len(colors)], linewidth=3)
        for whisker in bp["whiskers"]:
            whisker.set(color=colors[i % len(colors)], linewidth=2.5, linestyle=":")
        for cap in bp["caps"]:
            cap.set(color=colors[i % len(colors)], linewidth=3)
        for flier in bp["fliers"]:
            flier.set(markeredgecolor=colors[i % len(colors)],
                      markerfacecolor=colors[i % len(colors)], alpha=0.75)
        ax.set_xlabel("Scaling", fontsize=12)
        ax.set_ylabel("P[Type 1 Error] (%)", fontsize=12)
        ax.set_title(f"Batch Size: {n_paths}", fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # Hide unused subplots
    for j in range(len(n_paths_list), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(title, fontsize=14, y=1.0)
    plt.subplots_adjust(hspace=0.3)
    if filename:
        plt.savefig(filename, bbox_inches="tight", format="svg")
    plt.close()


# ---------------------------------------------------------------------------
# Core per-scenario analysis
# ---------------------------------------------------------------------------

def plot_sample_paths(h0_paths, h1_paths, title, filename, n_plot=5):
    label_first = True
    for p0, p1 in zip(h0_paths[:n_plot].cpu(), h1_paths[:n_plot].cpu()):
        plt.plot(p0[:, 1], p0[:, 0] - p0[0, 0], color="dodgerblue", alpha=0.75,
                 label=r"$\mathcal{H}_0$" if label_first else "")
        plt.plot(p1[:, 1], p1[:, 0] - p1[0, 0], color="tomato", alpha=0.75,
                 label=r"$\mathcal{H}_1$" if label_first else "")
        label_first = False
    plt.legend()
    make_grid()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", format="svg")
    plt.close()


def run_scenario_analysis(
    h0_paths, h1_paths, label, save_dir,
    path_bank_size,
    sig_kernel,
    n_atoms=500, n_paths=128, alpha=0.05,
    n_atoms_lvl=2048, n_paths_lvl=128, ks=None,
    scalings=None, n_paths_err_list=None,
    n_atoms_err=100, num_sim=100,
):
    """
    Full standard analysis for one (H0, H1) pair:
      - Unbiased MMD distribution
      - Level contributions
      - Type 1 / Type 2 errors vs scaling and batch size
    """
    if ks is None:
        ks = [1, 2, 3, 4]
    if scalings is None:
        scalings = np.linspace(0, 5, 20)
    if n_paths_err_list is None:
        n_paths_err_list = [20, 40, 60, 120]

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/data", exist_ok=True)

    # Two-sample test
    print(f"  [{label}] Two-sample test...")
    h0_d, h1_d = return_mmd_distributions(
        h0_paths, h1_paths, sig_kernel.compute_mmd,
        n_atoms=n_atoms, batch_size=n_paths, estimator="ub",
    )
    h0_d = filter_nan(h0_d, tag=f"{label}/h0")
    h1_d = filter_nan(h1_d, tag=f"{label}/h1")
    if h0_d and h1_d:
        plot_dist(h0_d, h1_d, len(h0_d), alpha,
                  f"{save_dir}/mmd_{label}_unbiased.svg", svg=True)
        plt.close()

    # Level contributions
    print(f"  [{label}] Level contributions...")
    h0_Mk, h1_Mk = get_level_values(h0_paths, h1_paths, n_atoms_lvl, n_paths_lvl, ks, path_bank_size)
    h0_Mk = np.asarray(h0_Mk)
    h1_Mk = np.asarray(h1_Mk)
    plot_level_contributions(h0_Mk, h1_Mk, n_atoms_lvl, ks,
                              f"{save_dir}/levels_{label}.svg",
                              svg=True, scientific=True, filter=False)
    plt.close()

    # Errors vs scaling & batch size
    print(f"  [{label}] Errors vs scaling ({num_sim} sims)...")
    type1_list, type2_list = generate_error_probs_linear_kernel(
        sig_kernel, h0_paths, h1_paths,
        n_atoms_err, n_paths_err_list, alpha, scalings,
        "ub", num_sim, device,
        filename=f"hawkes_{label}", folder=f"{save_dir}/data/",
    )

    plot_type2_error_n(type2_list, scalings, n_paths_err_list, num_sim,
                       title=f"{label} — Type 2 Error vs Scaling",
                       filename=f"{save_dir}/type2_{label}.svg")

    plot_type1_error_n(type1_list, scalings, n_paths_err_list, num_sim,
                       title=f"{label} — Type 1 Error vs Scaling",
                       filename=f"{save_dir}/type1_{label}.svg")

    return type1_list, type2_list


# ---------------------------------------------------------------------------
# Parameter-sweep analysis (errors vs delta)
# ---------------------------------------------------------------------------

def compute_errors_vs_delta(delta_vals, make_paths_fn, n_atoms, n_paths, alpha,
                             num_rep, sig_kernel, fixed_scaling=1.0, desc="sweep"):
    """
    For each delta in delta_vals, call make_paths_fn(delta) -> (h0, h1),
    compute Type 1 / Type 2 errors and repeat num_rep times.
    Returns dicts {delta -> [values over reps]}.
    """
    type1_results = defaultdict(list)
    type2_results = defaultdict(list)

    for _ in tqdm(range(num_rep), desc=desc):
        for delta in delta_vals:
            h0, h1 = make_paths_fn(delta)
            t2e, t1e = get_type1_type2_errors(
                sig_kernel, h0, h1,
                scaling=fixed_scaling,
                n_atoms=n_atoms, n_paths=n_paths,
                estimator="ub", alpha=alpha, device=device,
            )
            type1_results[delta].append(100.0 - float(t1e))
            type2_results[delta].append(float(t2e))

    return type1_results, type2_results


def plot_errors_vs_delta(type1_results, type2_results, delta_vals,
                          x_label, title_suffix="", filename_prefix="hawkes"):
    """Plot Type 1 and Type 2 errors (mean ± std) as a function of a parameter delta."""
    dl_arr = np.array(delta_vals)

    t1_mean = np.array([np.mean(type1_results[d]) for d in delta_vals])
    t1_std  = np.array([np.std(type1_results[d])  for d in delta_vals])
    t2_mean = np.array([np.mean(type2_results[d]) for d in delta_vals])
    t2_std  = np.array([np.std(type2_results[d])  for d in delta_vals])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mean, std, ylabel, color in zip(
        axes,
        [t1_mean, t2_mean],
        [t1_std, t2_std],
        ["Type 1 Error (%)", "Type 2 Error (%)"],
        ["tomato", "dodgerblue"],
    ):
        ax.plot(dl_arr, mean, color=color, linewidth=2)
        ax.fill_between(dl_arr, mean - std, mean + std,
                         color=color, alpha=0.3, edgecolor="none")
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f"{ylabel} — {title_suffix}", fontsize=13)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True, color="black", alpha=0.15)
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_errors_vs_delta.svg", bbox_inches="tight", format="svg")
    plt.close()


# ===========================================================================
# Experiment configurations
# ===========================================================================
# Each config is a dict with simulation params + labelling info.
# "normalize" is always True; the two configs differ by T and param scale.

EXPERIMENT_CONFIGS = [
    # --- A: high-rate params, T=1, paths naturally O(1) after normalisation ---
    dict(
        name        = "strong_T1",
        T           = 1,
        grid_points = 100,
        # Scenario params
        S1_mu=5.0,  S1_alpha0=0.5, S1_alpha1=0.8,
        S2_mu0=5.0, S2_mu1=8.0,   S2_alpha=0.5,
        S3_mu0=5.0, S3_alpha0=0.5, S3_mu1=8.0, S3_alpha1=0.8,
        S4_mu=5.0,  S4_alpha=0.5,  S4_beta0=3.0, S4_beta1=8.0,
        fixed_beta  = 5.0,
        # Sensitivity sweep ranges
        delta_alphas = np.linspace(0, 0.6, 15),
        delta_mus    = np.linspace(0, 5.0, 15),
        joint_deltas = np.linspace(0, 1.0, 15),
        delta_mus_sens   = np.linspace(0, 6.0, 20),
        delta_betas_sens = np.linspace(0, 10.0, 20),
        sens_mu0=5.0, sens_alpha=0.5, sens_beta=5.0, sens_beta0=3.0,
    ),
    # --- B: standard params, T=20, normalised to O(1) ---
    dict(
        name        = "normal_T20",
        T           = 20,
        grid_points = 100,
        S1_mu=2.0,  S1_alpha0=0.3, S1_alpha1=0.6,
        S2_mu0=2.0, S2_mu1=3.0,   S2_alpha=0.3,
        S3_mu0=2.0, S3_alpha0=0.3, S3_mu1=3.0, S3_alpha1=0.6,
        S4_mu=2.0,  S4_alpha=0.3,  S4_beta0=3.0, S4_beta1=7.0,
        fixed_beta  = 5.0,
        delta_alphas = np.linspace(0, 0.7, 15),
        delta_mus    = np.linspace(0, 3.0, 15),
        joint_deltas = np.linspace(0, 1.0, 15),
        delta_mus_sens   = np.linspace(0, 4.0, 20),
        delta_betas_sens = np.linspace(0, 10.0, 20),
        sens_mu0=2.0, sens_alpha=0.3, sens_beta=5.0, sens_beta0=3.0,
    ),
]

# ===========================================================================
# Shared analysis hyper-parameters
# ===========================================================================
path_bank_size   = 10000
alpha_test       = 0.05
scalings_sweep   = np.linspace(0.1, 5, 10)   # removed 0 point (0 not discriminative)
n_paths_err_list = [20, 60, 120]            # dropped 40 (4 -> 3 batch sizes)
n_atoms_err      = 50                       # 100 -> 50
num_sim          = 25                       # 100 -> 25
n_atoms_delta    = 100                      # 200 -> 100
n_paths_delta    = 64
num_rep_delta    = 15                       # 20 -> 15

os.makedirs(DATA_DIR, exist_ok=True)

# ===========================================================================
# Main loop: experiment config × kernel
# ===========================================================================
for cfg in EXPERIMENT_CONFIGS:
    exp_name   = cfg["name"]
    T          = cfg["T"]
    grid_points = cfg["grid_points"]
    fixed_beta = cfg["fixed_beta"]

    print(f"\n{'#'*65}")
    print(f"  EXPERIMENT: {exp_name}  (T={T}, grid_points={grid_points})")
    print(f"{'#'*65}")

    # Pre-simulate path banks once per config (shared across kernels)
    h0_s1, h1_s1 = load_hawkes_paths(
        cfg["S1_mu"], cfg["S1_alpha0"], cfg["S1_mu"], cfg["S1_alpha1"],
        fixed_beta, path_bank_size, grid_points, T)

    h0_s2, h1_s2 = load_hawkes_paths(
        cfg["S2_mu0"], cfg["S2_alpha"], cfg["S2_mu1"], cfg["S2_alpha"],
        fixed_beta, path_bank_size, grid_points, T)

    h0_s3, h1_s3 = load_hawkes_paths(
        cfg["S3_mu0"], cfg["S3_alpha0"], cfg["S3_mu1"], cfg["S3_alpha1"],
        fixed_beta, path_bank_size, grid_points, T)

    h0_s4, h1_s4 = load_hawkes_paths(
        cfg["S4_mu"], cfg["S4_alpha"], cfg["S4_mu"], cfg["S4_alpha"],
        cfg["S4_beta0"], path_bank_size, grid_points, T, beta1=cfg["S4_beta1"])

    # Sample path plots (kernel-independent)
    base_dir = f"{DATA_DIR}/{exp_name}"
    os.makedirs(base_dir, exist_ok=True)

    # Write per-experiment metadata file
    meta = {
        "experiment": exp_name,
        "run_date": datetime.datetime.now().isoformat(),
        "simulation": {
            "T": T,
            "grid_points": grid_points,
            "path_bank_size": path_bank_size,
            "fixed_beta": fixed_beta,
        },
        "analysis": {
            "alpha_test": alpha_test,
            "scalings_sweep": list(scalings_sweep),
            "n_paths_err_list": n_paths_err_list,
            "n_atoms_err": n_atoms_err,
            "num_sim": num_sim,
            "n_atoms_delta": n_atoms_delta,
            "n_paths_delta": n_paths_delta,
            "num_rep_delta": num_rep_delta,
        },
        "kernels": [k for k, _ in KERNELS],
        "scenarios": {
            "S1_same_mu_diff_alpha": {
                "H0": {"mu": cfg["S1_mu"], "alpha": cfg["S1_alpha0"], "beta": fixed_beta},
                "H1": {"mu": cfg["S1_mu"], "alpha": cfg["S1_alpha1"], "beta": fixed_beta},
                "delta_alphas": np.asarray(cfg["delta_alphas"]).tolist(),
            },
            "S2_diff_mu_same_alpha": {
                "H0": {"mu": cfg["S2_mu0"], "alpha": cfg["S2_alpha"], "beta": fixed_beta},
                "H1": {"mu": cfg["S2_mu1"], "alpha": cfg["S2_alpha"], "beta": fixed_beta},
                "delta_mus": np.asarray(cfg["delta_mus"]).tolist(),
            },
            "S3_diff_mu_diff_alpha": {
                "H0": {"mu": cfg["S3_mu0"], "alpha": cfg["S3_alpha0"], "beta": fixed_beta},
                "H1": {"mu": cfg["S3_mu1"], "alpha": cfg["S3_alpha1"], "beta": fixed_beta},
                "joint_deltas": np.asarray(cfg["joint_deltas"]).tolist(),
            },
            "S4_diff_beta": {
                "H0": {"mu": cfg["S4_mu"], "alpha": cfg["S4_alpha"], "beta": cfg["S4_beta0"]},
                "H1": {"mu": cfg["S4_mu"], "alpha": cfg["S4_alpha"], "beta": cfg["S4_beta1"]},
            },
            "sensitivity_delta_mu": {
                "H0": {"mu": cfg["sens_mu0"], "alpha": cfg["sens_alpha"], "beta": cfg["sens_beta"]},
                "H1_varies": "mu = mu0 + delta_mu",
                "delta_mus_sens": np.asarray(cfg["delta_mus_sens"]).tolist(),
            },
            "sensitivity_delta_beta": {
                "H0": {"mu": cfg["sens_mu0"], "alpha": cfg["sens_alpha"], "beta": cfg["sens_beta0"]},
                "H1_varies": "beta = beta0 + delta_beta",
                "delta_betas_sens": np.asarray(cfg["delta_betas_sens"]).tolist(),
            },
        },
    }
    meta_path = f"{base_dir}/metadata.json"
    with open(meta_path, "w") as _mf:
        json.dump(meta, _mf, indent=2)
    print(f"  Metadata written to {meta_path}")

    plot_sample_paths(h0_s1, h1_s1,
        title=f"[{exp_name}] S1 — same mu, diff alpha",
        filename=f"{base_dir}/sample_paths_s1.svg")
    plot_sample_paths(h0_s2, h1_s2,
        title=f"[{exp_name}] S2 — diff mu, same alpha",
        filename=f"{base_dir}/sample_paths_s2.svg")
    plot_sample_paths(h0_s3, h1_s3,
        title=f"[{exp_name}] S3 — diff mu & alpha",
        filename=f"{base_dir}/sample_paths_s3.svg")
    plot_sample_paths(h0_s4, h1_s4,
        title=f"[{exp_name}] S4 — diff beta",
        filename=f"{base_dir}/sample_paths_s4.svg")

    for kernel_name, sig_kernel in KERNELS:
        print(f"\n{'='*65}")
        print(f"  {exp_name}  |  kernel={kernel_name}")
        print(f"{'='*65}")

        kdir = f"{base_dir}/{kernel_name}"

        # ---------------------------------------------------------------
        # SCENARIO 1 — same mu, different alpha
        # ---------------------------------------------------------------
        run_scenario_analysis(
            h0_s1, h1_s1,
            label=f"s1_same_mu_diff_alpha",
            save_dir=f"{kdir}/scenario1",
            path_bank_size=path_bank_size,
            sig_kernel=sig_kernel,
            scalings=scalings_sweep, n_paths_err_list=n_paths_err_list,
            n_atoms_err=n_atoms_err, num_sim=num_sim,
        )

        delta_alphas = cfg["delta_alphas"]
        S1_mu, S1_alpha0 = cfg["S1_mu"], cfg["S1_alpha0"]

        def make_s1_paths(d_alpha, _mu=S1_mu, _a0=S1_alpha0, _beta=fixed_beta):
            return load_hawkes_paths(_mu, _a0, _mu, _a0 + d_alpha,
                                     _beta, path_bank_size, grid_points, T)

        t1_s1, t2_s1 = compute_errors_vs_delta(
            delta_alphas, make_s1_paths,
            n_atoms=n_atoms_delta, n_paths=n_paths_delta,
            alpha=alpha_test, num_rep=num_rep_delta,
            sig_kernel=sig_kernel,
            desc=f"{exp_name}/{kernel_name} S1: Δalpha",
        )
        plot_errors_vs_delta(
            t1_s1, t2_s1, delta_alphas,
            x_label=r"$\alpha_1 - \alpha_0$",
            title_suffix=f"[{exp_name}/{kernel_name}] Same mu",
            filename_prefix=f"{kdir}/scenario1/hawkes_s1",
        )

        # ---------------------------------------------------------------
        # SCENARIO 2 — different mu, same alpha
        # ---------------------------------------------------------------
        run_scenario_analysis(
            h0_s2, h1_s2,
            label=f"s2_diff_mu_same_alpha",
            save_dir=f"{kdir}/scenario2",
            path_bank_size=path_bank_size,
            sig_kernel=sig_kernel,
            scalings=scalings_sweep, n_paths_err_list=n_paths_err_list,
            n_atoms_err=n_atoms_err, num_sim=num_sim,
        )

        delta_mus = cfg["delta_mus"]
        S2_mu0, S2_alpha = cfg["S2_mu0"], cfg["S2_alpha"]

        def make_s2_paths(d_mu, _mu0=S2_mu0, _alpha=S2_alpha, _beta=fixed_beta):
            return load_hawkes_paths(_mu0, _alpha, _mu0 + d_mu, _alpha,
                                     _beta, path_bank_size, grid_points, T)

        t1_s2, t2_s2 = compute_errors_vs_delta(
            delta_mus, make_s2_paths,
            n_atoms=n_atoms_delta, n_paths=n_paths_delta,
            alpha=alpha_test, num_rep=num_rep_delta,
            sig_kernel=sig_kernel,
            desc=f"{exp_name}/{kernel_name} S2: Δmu",
        )
        plot_errors_vs_delta(
            t1_s2, t2_s2, delta_mus,
            x_label=r"$\mu_1 - \mu_0$",
            title_suffix=f"[{exp_name}/{kernel_name}] Same alpha",
            filename_prefix=f"{kdir}/scenario2/hawkes_s2",
        )

        # ---------------------------------------------------------------
        # SCENARIO 3 — different mu AND different alpha
        # ---------------------------------------------------------------
        run_scenario_analysis(
            h0_s3, h1_s3,
            label=f"s3_diff_mu_diff_alpha",
            save_dir=f"{kdir}/scenario3",
            path_bank_size=path_bank_size,
            sig_kernel=sig_kernel,
            scalings=scalings_sweep, n_paths_err_list=n_paths_err_list,
            n_atoms_err=n_atoms_err, num_sim=num_sim,
        )

        joint_deltas = cfg["joint_deltas"]
        S3_mu0    = float(cfg["S3_mu0"])
        S3_alpha0 = float(cfg["S3_alpha0"])
        S3_mu1    = float(cfg["S3_mu1"])
        S3_alpha1 = float(cfg["S3_alpha1"])

        def make_s3_paths(delta,
                           _mu0=S3_mu0, _a0=S3_alpha0,
                           _mu1=S3_mu1, _a1=S3_alpha1, _beta=fixed_beta):
            mu_cur    = _mu0 + delta * (_mu1 - _mu0)
            alpha_cur = _a0  + delta * (_a1  - _a0)
            return load_hawkes_paths(_mu0, _a0, mu_cur, alpha_cur,
                                     _beta, path_bank_size, grid_points, T)

        t1_s3, t2_s3 = compute_errors_vs_delta(
            joint_deltas, make_s3_paths,
            n_atoms=n_atoms_delta, n_paths=n_paths_delta,
            alpha=alpha_test, num_rep=num_rep_delta,
            sig_kernel=sig_kernel,
            desc=f"{exp_name}/{kernel_name} S3: joint δ",
        )
        plot_errors_vs_delta(
            t1_s3, t2_s3, joint_deltas,
            x_label=r"Parameter interpolation $\delta$",
            title_suffix=f"[{exp_name}/{kernel_name}] Diff mu & alpha",
            filename_prefix=f"{kdir}/scenario3/hawkes_s3",
        )

        # ---------------------------------------------------------------
        # SCENARIO 4 — different beta
        # ---------------------------------------------------------------
        run_scenario_analysis(
            h0_s4, h1_s4,
            label=f"s4_diff_beta",
            save_dir=f"{kdir}/scenario4",
            path_bank_size=path_bank_size,
            sig_kernel=sig_kernel,
            scalings=scalings_sweep, n_paths_err_list=n_paths_err_list,
            n_atoms_err=n_atoms_err, num_sim=num_sim,
        )

        # ---------------------------------------------------------------
        # SENSITIVITY Δmu
        # ---------------------------------------------------------------
        sens_mu0   = cfg["sens_mu0"]
        sens_alpha = cfg["sens_alpha"]
        sens_beta  = cfg["sens_beta"]
        delta_mus_sens = cfg["delta_mus_sens"]

        def make_sens_mu_paths(d_mu, _mu0=sens_mu0, _alpha=sens_alpha, _beta=sens_beta):
            return load_hawkes_paths(_mu0, _alpha, _mu0 + d_mu, _alpha,
                                     _beta, path_bank_size, grid_points, T)

        os.makedirs(f"{kdir}/sensitivity", exist_ok=True)
        t1_sm, t2_sm = compute_errors_vs_delta(
            delta_mus_sens, make_sens_mu_paths,
            n_atoms=n_atoms_delta, n_paths=n_paths_delta,
            alpha=alpha_test, num_rep=num_rep_delta,
            sig_kernel=sig_kernel,
            desc=f"{exp_name}/{kernel_name} sens Δmu",
        )
        plot_errors_vs_delta(
            t1_sm, t2_sm, delta_mus_sens,
            x_label=r"$\mu_1 - \mu_0$",
            title_suffix=f"[{exp_name}/{kernel_name}] Same alpha & beta",
            filename_prefix=f"{kdir}/sensitivity/hawkes_sens_mu",
        )

        # ---------------------------------------------------------------
        # SENSITIVITY Δbeta
        # ---------------------------------------------------------------
        sens_beta0     = cfg["sens_beta0"]
        delta_betas_sens = cfg["delta_betas_sens"]

        def make_sens_beta_paths(d_beta, _mu0=sens_mu0, _alpha=sens_alpha, _b0=sens_beta0):
            return load_hawkes_paths(_mu0, _alpha, _mu0, _alpha,
                                     _b0, path_bank_size, grid_points, T,
                                     beta1=_b0 + d_beta)

        t1_sb, t2_sb = compute_errors_vs_delta(
            delta_betas_sens, make_sens_beta_paths,
            n_atoms=n_atoms_delta, n_paths=n_paths_delta,
            alpha=alpha_test, num_rep=num_rep_delta,
            sig_kernel=sig_kernel,
            desc=f"{exp_name}/{kernel_name} sens Δbeta",
        )
        plot_errors_vs_delta(
            t1_sb, t2_sb, delta_betas_sens,
            x_label=r"$\beta_1 - \beta_0$",
            title_suffix=f"[{exp_name}/{kernel_name}] Same mu & alpha",
            filename_prefix=f"{kdir}/sensitivity/hawkes_sens_beta",
        )

# ===========================================================================
# Write summary report
# ===========================================================================
summary_path = f"{DATA_DIR}/analysis_summary.txt"
with open(summary_path, "w") as f:
    f.write(f"Hawkes Analysis Summary\n")
    f.write(f"Run date: {datetime.datetime.now().isoformat()}\n")
    f.write(f"Experiments: {[c['name'] for c in EXPERIMENT_CONFIGS]}\n")
    f.write(f"Kernels: {[k for k, _ in KERNELS]}\n")
    f.write(f"path_bank_size={path_bank_size}, num_sim={num_sim}, "
            f"num_rep_delta={num_rep_delta}\n")
    f.write("\n--- NaN / non-finite MMD values ---\n")
    total_removed = 0
    for entry in nan_log:
        if entry["removed"] > 0:
            f.write(f"  {entry['tag']}: {entry['removed']}/{entry['total']} removed\n")
            total_removed += entry["removed"]
    if total_removed == 0:
        f.write("  None — all MMD values were finite.\n")
    else:
        f.write(f"  TOTAL removed: {total_removed}\n")
print(f"\nSummary written to {summary_path}")

print("\n" + "="*65)
print(f"All analyses complete. Results saved in {DATA_DIR}/")
print("="*65)
