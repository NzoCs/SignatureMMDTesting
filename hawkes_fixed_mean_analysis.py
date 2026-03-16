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

import os

import matplotlib

matplotlib.use("Agg")  # headless cluster execution
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from src.utils.helper_functions.plot_helper_functions import (
    make_grid,
)
from src.utils.plotting_functions import (
    plot_dist,
    plot_level_contributions,
)
from src.mmd.distribution_functions import (
    return_mmd_distributions,
    get_level_values,
    generate_error_probs_linear_kernel,
    get_type1_type2_errors,
)
from src.mmd.mmd import SigKernel, LinearKernel

# ---------------------------------------------------------------------------
# Device & kernels
# ---------------------------------------------------------------------------
# Output directory for analysis results (change here to adjust output folder)
DATA_DIR = "eq_hawkes_fixed_mean"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

dyadic_order = 0
sig_kernel_linear = SigKernel(static_kernel=LinearKernel(), dyadic_order=dyadic_order)

KERNELS = [("linear", sig_kernel_linear)]

# Module-level alias kept for any internal helper that may reference it
# signature_kernel = sig_kernel_rbf

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

    def __init__(
        self, mu, alpha, beta, dim_process, start_time=0.0, end_time=1.0, seed=None
    ):
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


def sim_hawkes_model(mu, alpha, beta, num_sim, num_time_steps, T, burn_in=100.0):
    """
    Simulate Hawkes process count paths on [0, T] using Ogata's thinning algorithm.
    Intensity: λ(t) = μ + α * Σ_{t_i < t} exp(-β * (t - t_i))

    Includes a burn_in period to reach stationary regime.
    Returns array of shape (num_time_steps, num_sim, 2) — (count, time).
    """
    time_grid = np.linspace(0, T, num_time_steps)
    paths = np.zeros((num_time_steps, num_sim))

    for s in range(num_sim):
        simulator = HawkesSimulator(
            mu=[mu],
            alpha=[[alpha]],
            beta=[[beta]],
            dim_process=1,
            start_time=0.0,
            end_time=T + burn_in,
        )
        event_times, _ = simulator.simulate()

        # Discard burn_in period and shift times
        event_times = event_times[event_times >= burn_in] - burn_in

        if len(event_times) > 0:
            paths[:, s] = np.searchsorted(event_times, time_grid, side="right")

    return np.concatenate(
        (
            paths[:, :, None],
            np.repeat(time_grid[:, None, None], repeats=num_sim, axis=1),
        ),
        axis=2,
    )


def load_hawkes_paths(
    mu0,
    alpha0,
    mu1,
    alpha1,
    beta0,
    path_bank_size,
    grid_points,
    T,
    beta1=None,
    normalize=True,
    burn_in=100.0,
):
    """Simulate and return centred (and optionally normalised) torch path banks.

    If normalize=True:
      - count channel divided by empirical std of final count (H0 bank)
      - time  channel divided by T  (so time lives in [0, 1])
    This ensures both channels are O(1) regardless of T or event-rate.
    beta1 defaults to beta0 when omitted.
    """
    if beta1 is None:
        beta1 = beta0
    h0_bank = sim_hawkes_model(
        mu0, alpha0, beta0, path_bank_size, grid_points, T, burn_in=burn_in
    )
    h1_bank = sim_hawkes_model(
        mu1, alpha1, beta1, path_bank_size, grid_points, T, burn_in=burn_in
    )

    h0 = torch.transpose(torch.from_numpy(h0_bank), 0, 1).to(
        device=device, dtype=torch.float32
    )
    h1 = torch.transpose(torch.from_numpy(h1_bank), 0, 1).to(
        device=device, dtype=torch.float32
    )

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
        print(
            f"  Warning: removed {n_removed}/{len(dists)} non-finite MMD values. [{tag}]"
        )
    return clean


def plot_type2_error_n(
    type2_list, scalings, n_paths_list, num_sim, title="", filename=None, colors=None
):
    """plot_type2_error with configurable num_sim (not hardcoded to 100)."""
    if colors is None:
        colors = ["magenta", "green", "darkorange", "blue"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, n_paths in enumerate(n_paths_list):
        t2e = [type2_list[j][n_paths] for j in range(num_sim)]
        t2e_mean = np.mean(np.asarray(t2e), axis=0)
        t2e_std = np.std(np.asarray(t2e), axis=0)
        ax.plot(scalings, t2e_mean, alpha=1, label=f"{n_paths}", color=colors[i])
        ax.fill_between(
            scalings,
            t2e_mean - t2e_std,
            t2e_mean + t2e_std,
            color=colors[i],
            alpha=0.3,
            edgecolor="none",
        )
    ax.set_ylabel("P[Type 2 Error] (%)", fontsize=13)
    ax.set_xlabel("Scaling", fontsize=13)
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=13)
    ax.grid(True, color="black", alpha=0.2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.title(title, fontsize=13)
    if filename:
        plt.savefig(filename, bbox_inches="tight", format="svg")
    plt.close()


def plot_type1_error_n(
    type1_list, scalings, n_paths_list, num_sim, title="", filename=None, colors=None
):
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
        bp = ax.boxplot(
            np.asarray(t1e)[:, 1::2],
            patch_artist=True,
            labels=np.round(scalings[1::2], 2),
        )
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
            flier.set(
                markeredgecolor=colors[i % len(colors)],
                markerfacecolor=colors[i % len(colors)],
                alpha=0.75,
            )
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
        plt.plot(
            p0[:, 1],
            p0[:, 0] - p0[0, 0],
            color="dodgerblue",
            alpha=0.75,
            label=r"$\mathcal{H}_0$" if label_first else "",
        )
        plt.plot(
            p1[:, 1],
            p1[:, 0] - p1[0, 0],
            color="tomato",
            alpha=0.75,
            label=r"$\mathcal{H}_1$" if label_first else "",
        )
        label_first = False
    plt.legend()
    make_grid()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", format="svg")
    plt.close()


def run_scenario_analysis(
    h0_paths,
    h1_paths,
    label,
    save_dir,
    path_bank_size,
    sig_kernel,
    n_atoms=500,
    n_paths=128,
    alpha=0.05,
    n_atoms_lvl=2048,
    n_paths_lvl=128,
    ks=None,
    scalings=None,
    n_paths_err_list=None,
    n_atoms_err=100,
    num_sim=100,
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
        h0_paths,
        h1_paths,
        sig_kernel.compute_mmd,
        n_atoms=n_atoms,
        batch_size=n_paths,
        estimator="ub",
    )
    h0_d = filter_nan(h0_d, tag=f"{label}/h0")
    h1_d = filter_nan(h1_d, tag=f"{label}/h1")
    if h0_d and h1_d:
        plot_dist(
            h0_d,
            h1_d,
            len(h0_d),
            alpha,
            f"{save_dir}/mmd_{label}_unbiased.svg",
            svg=True,
        )
        plt.close()

    # Level contributions
    print(f"  [{label}] Level contributions...")
    h0_Mk, h1_Mk = get_level_values(
        h0_paths, h1_paths, n_atoms_lvl, n_paths_lvl, ks, path_bank_size
    )
    h0_Mk = np.asarray(h0_Mk)
    h1_Mk = np.asarray(h1_Mk)
    plot_level_contributions(
        h0_Mk,
        h1_Mk,
        n_atoms_lvl,
        ks,
        f"{save_dir}/levels_{label}.svg",
        svg=True,
        scientific=True,
        filter=False,
    )
    plt.close()

    # Errors vs scaling & batch size
    print(f"  [{label}] Errors vs scaling ({num_sim} sims)...")
    type1_list, type2_list = generate_error_probs_linear_kernel(
        sig_kernel,
        h0_paths,
        h1_paths,
        n_atoms_err,
        n_paths_err_list,
        alpha,
        scalings,
        "ub",
        num_sim,
        device,
        filename=f"hawkes_{label}",
        folder=f"{save_dir}/data/",
    )

    plot_type2_error_n(
        type2_list,
        scalings,
        n_paths_err_list,
        num_sim,
        title=f"{label} — Type 2 Error vs Scaling",
        filename=f"{save_dir}/type2_{label}.svg",
    )

    plot_type1_error_n(
        type1_list,
        scalings,
        n_paths_err_list,
        num_sim,
        title=f"{label} — Type 1 Error vs Scaling",
        filename=f"{save_dir}/type1_{label}.svg",
    )

    return type1_list, type2_list


# ---------------------------------------------------------------------------
# Parameter-sweep analysis (errors vs delta)
# ---------------------------------------------------------------------------


def compute_errors_vs_delta(
    delta_vals,
    make_paths_fn,
    n_atoms,
    n_paths,
    alpha,
    num_rep,
    sig_kernel,
    fixed_scaling=1.0,
    desc="sweep",
):
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
                sig_kernel,
                h0,
                h1,
                scaling=fixed_scaling,
                n_atoms=n_atoms,
                n_paths=n_paths,
                estimator="ub",
                alpha=alpha,
                device=device,
            )
            type1_results[delta].append(100.0 - float(t1e))
            type2_results[delta].append(float(t2e))

    return type1_results, type2_results


def plot_errors_vs_delta(
    type1_results,
    type2_results,
    delta_vals,
    x_label,
    title_suffix="",
    filename_prefix="hawkes",
):
    """Plot Type 1 and Type 2 errors (mean ± std) as a function of a parameter delta."""
    dl_arr = np.array(delta_vals)

    t1_mean = np.array([np.mean(type1_results[d]) for d in delta_vals])

    if np.mean(t1_mean) > 10.0:
        print(
            f"Skipping plot {filename_prefix}: average Type 1 error ({np.mean(t1_mean):.2f}%) > 10%"
        )
        return

    t1_std = np.array([np.std(type1_results[d]) for d in delta_vals])
    t2_mean = np.array([np.mean(type2_results[d]) for d in delta_vals])
    t2_std = np.array([np.std(type2_results[d]) for d in delta_vals])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mean, std, ylabel, color in zip(
        axes,
        [t1_mean, t2_mean],
        [t1_std, t2_std],
        ["Type 1 Error (%)", "Type 2 Error (%)"],
        ["tomato", "dodgerblue"],
    ):
        ax.plot(dl_arr, mean, color=color, linewidth=2)
        ax.fill_between(
            dl_arr, mean - std, mean + std, color=color, alpha=0.3, edgecolor="none"
        )
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f"{ylabel} — {title_suffix}", fontsize=13)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True, color="black", alpha=0.15)
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)

    plt.tight_layout()
    plt.savefig(
        f"{filename_prefix}_errors_vs_delta.svg", bbox_inches="tight", format="svg"
    )
    plt.close()


# ===========================================================================
# Experiment configurations for Fixed Mean: Lambda = 100
# ===========================================================================
def main_experiment():
    # Mean equation: Lambda = mu / (1 - alpha/beta)
    # Let beta = 10. For Lambda = 100: mu = 100 - 10*alpha

    import json
    import os
    import numpy as np

    # Override DATA_DIR
    DATA_DIR = "eq_hawkes_fixed_mean"
    os.makedirs(DATA_DIR, exist_ok=True)

    T = 1
    grid_points = 100
    fixed_beta = 10.0
    target_mean = 100.0

    path_bank_size = 10000
    alpha_test = 0.05
    scalings_sweep = np.logspace(-1, np.log10(5), 20)
    n_paths_err_list = [120, 480]
    n_atoms_err = 50
    num_sim = 25

    # Setup H0 (baseline)
    alpha0 = 5.0
    mu0 = 100.0 - 10.0 * alpha0
    h0_mean = mu0 / (1.0 - alpha0 / fixed_beta)

    print(f"\n{'=' * 65}")
    print(f"EXPERIMENT: Fixed Mean (Lambda = {target_mean})")
    print(f"H0: mu={mu0}, alpha={alpha0}, beta={fixed_beta} -> Lambda = {h0_mean}")
    print(f"{'=' * 65}\n")

    # H1 Sweep (Avoid exactly alpha0 so it's not the null hypothesis every time, though we can include it to test Type 1)
    alphas_h1 = np.linspace(2, 8, 7)  # [2, 3, 4, 5, 6, 7, 8]
    mus_h1 = 100.0 - 10.0 * alphas_h1

    for kernel_name, sig_kernel in KERNELS:
        print(f"\n--- Kernel: {kernel_name} ---")

        # We will compute Type 1 / Type 2 error for each alpha in H1 vs H0
        # and plot error vs parameter (delta alpha)

        def make_fixed_mean_paths(alpha_1, _mu0=mu0, _a0=alpha0, _beta=fixed_beta):
            mu_1 = 100.0 - 10.0 * alpha_1
            h0, h1 = load_hawkes_paths(
                _mu0,
                _a0,
                mu_1,
                alpha_1,
                _beta,
                path_bank_size=n_paths_delta
                * 2,  # just enough for the evaluation batch
                grid_points=grid_points,
                T=T,
                beta1=_beta,
                normalize=True,
            )
            return h0, h1

        save_dir = f"{DATA_DIR}/{kernel_name}"
        os.makedirs(save_dir, exist_ok=True)

        # Run parameter sweep first
        print(f"  [Parameter Sweep] alphas in {alphas_h1}")
        n_atoms_delta = 100
        n_paths_delta = 64
        num_rep_delta = 15

        t1_res, t2_res = compute_errors_vs_delta(
            alphas_h1,
            make_fixed_mean_paths,
            n_atoms=n_atoms_delta,
            n_paths=n_paths_delta,
            alpha=alpha_test,
            num_rep=num_rep_delta,
            sig_kernel=sig_kernel,
            desc=f"Fixed Mean sweep: {kernel_name}",
        )

        plot_errors_vs_delta(
            t1_res,
            t2_res,
            alphas_h1,
            x_label=r"$\alpha_1$ (with $\mu_1 = 100 - 10\alpha_1$)",
            title_suffix=f"Fixed Mean (Lambda={target_mean}) Kernel={kernel_name}",
            filename_prefix=f"{save_dir}/sweep_fixed_mean",
        )

        # Evaluate which alpha gave the highest mean Type 2 error
        worst_alpha = alphas_h1[0]
        highest_t2e = -1.0
        for a in alphas_h1:
            mean_t2e = np.mean(t2_res[a])
            if mean_t2e > highest_t2e:
                highest_t2e = mean_t2e
                worst_alpha = a

        print(
            f"  -> Selected worst case alpha for detailed scenario: alpha={worst_alpha} (Avg T2E: {highest_t2e:.2f}%)"
        )

        # Detailed scenario on the worst parameter
        alpha_extreme = worst_alpha
        mu_extreme = 100.0 - 10.0 * alpha_extreme

        print(f"  [Detailed Scenario] H1: mu={mu_extreme}, alpha={alpha_extreme}")
        h0_s, h1_s = load_hawkes_paths(
            mu0,
            alpha0,
            mu_extreme,
            alpha_extreme,
            fixed_beta,
            path_bank_size,
            grid_points,
            T,
            beta1=fixed_beta,
            normalize=True,
        )

        run_scenario_analysis(
            h0_s,
            h1_s,
            label=f"fixed_mean_worst_case_alpha_{alpha_extreme}",
            save_dir=f"{save_dir}/detailed_scenario",
            path_bank_size=path_bank_size,
            sig_kernel=sig_kernel,
            scalings=scalings_sweep,
            n_paths_err_list=n_paths_err_list,
            n_atoms_err=n_atoms_err,
            num_sim=num_sim,
        )

    print("\n" + "=" * 65)
    print(f"Fixed mean analysis complete. Results saved in {DATA_DIR}/")
    print("=" * 65)

    # Save metadata
    metadata = {
        "experiment": "Fixed Mean Hawkes",
        "target_mean": target_mean,
        "fixed_beta": fixed_beta,
        "H0": {"mu0": mu0, "alpha0": alpha0, "mean": h0_mean},
        "H1_sweep_alphas": alphas_h1.tolist(),
        "H1_sweep_mus": mus_h1.tolist(),
        "selected_worst_alpha": worst_alpha,
        "scalings_sweep": scalings_sweep.tolist(),
        "n_paths_err_list": n_paths_err_list,
        "n_atoms_err": n_atoms_err,
        "num_sim": num_sim,
        "T": T,
        "grid_points": grid_points,
    }

    with open(f"{DATA_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main_experiment()
