"""
Poisson Process Analysis - Signature MMD Two-Sample Statistical Tests
Andrew Alden, Blanka Horvath, Zacharia Issa

Sections:
  1. Model Setup
  2. Two-Sample Hypothesis Test (Unbiased & Biased)
  3. Apply Scaling
  4. Level Contributions
  5. Errors vs Lambda Difference
  6. Errors vs Scaling and Number of Examples
"""

import math
import os
import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless cluster execution
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from collections import defaultdict
from tqdm import tqdm

from src.utils.helper_functions.plot_helper_functions import make_grid, golden_dimensions
from src.utils.helper_functions.global_helper_functions import get_project_root
from src.utils.plotting_functions import (
    plot_dist,
    plot_level_contributions,
    plot_type2_error,
    plot_type1_error,
    plot_aggregate_type1_error,
    plot_dist_boxen,
)
from src.mmd.distribution_functions import (
    return_mmd_distributions,
    expected_type2_error,
    get_level_values,
    generate_error_probs_linear_kernel,
    get_type1_type2_errors,
)
from src.mmd.level_functions import lambda_k, level_k_contribution, mmd_est_k, gramda_k
from src.mmd.signature_functions import get_level_k_signatures_from_paths
from src.mmd.mmd import SigKernel, RBFKernel, LinearKernel

# ---------------------------------------------------------------------------
# 1. Device & kernel setup
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

dyadic_order = 0
static_kernel = LinearKernel()
signature_kernel = SigKernel(static_kernel=static_kernel, dyadic_order=dyadic_order)

# ---------------------------------------------------------------------------
# 2. Poisson path simulator
# ---------------------------------------------------------------------------

def sim_poisson_model(lam, num_sim, num_time_steps, T):
    """Simulate Poisson process paths on [0, T]."""
    time_steps = np.linspace(0, T, num_time_steps)
    dt = T / (num_time_steps - 1)
    increments = np.random.poisson(lam * dt, size=(num_time_steps - 1, num_sim))
    paths = np.vstack([np.zeros((1, num_sim)), np.cumsum(increments, axis=0)])
    return np.concatenate(
        (paths[:, :, None],
         np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim, axis=1)),
        axis=2,
    )


def load_paths(lam_0, lam_1, path_bank_size, grid_points, T):
    """Simulate and return centred torch path banks for H0 and H1."""
    h0_bank = sim_poisson_model(lam_0, path_bank_size, grid_points, T)
    h1_bank = sim_poisson_model(lam_1, path_bank_size, grid_points, T)

    h0 = torch.transpose(torch.from_numpy(h0_bank), 0, 1).to(device=device, dtype=torch.float32)
    h1 = torch.transpose(torch.from_numpy(h1_bank), 0, 1).to(device=device, dtype=torch.float32)

    for i in range(path_bank_size):
        h0[i] = h0[i] - h0[i, 0, :]
        h1[i] = h1[i] - h1[i, 0, :]
    return h0, h1


# ---------------------------------------------------------------------------
# 3. Base parameters
# ---------------------------------------------------------------------------
lam_0 = 10.0
lam_1 = 12.0

T = 1
grid_points = 20
path_bank_size = 10000

h0_paths, h1_paths = load_paths(lam_0, lam_1, path_bank_size, grid_points, T)

# ---------------------------------------------------------------------------
# 4. Plot sample paths
# ---------------------------------------------------------------------------
n_plot_paths = 5
label_first = True
for p0, p1 in zip(h0_paths[:n_plot_paths].cpu(), h1_paths[:n_plot_paths].cpu()):
    plt.plot(p0[:, 1], p0[:, 0] - p0[0, 0], color="dodgerblue", alpha=0.75,
             label=r"$\mathcal{H}_0$" if label_first else "")
    plt.plot(p1[:, 1], p1[:, 0] - p1[0, 0], color="tomato", alpha=0.75,
             label=r"$\mathcal{H}_1$" if label_first else "")
    label_first = False
plt.legend()
make_grid()
plt.title(f"Sample paths — lam_0={lam_0}, lam_1={lam_1}")
plt.tight_layout()
plt.savefig("poisson_sample_paths.svg", bbox_inches="tight", format="svg")
plt.show()

# ---------------------------------------------------------------------------
# 5. Two-sample hypothesis test
# ---------------------------------------------------------------------------
n_atoms = 500
n_paths = 128
alpha = 0.05

# -- Unbiased --
print("Computing unbiased MMD distributions...")
h0_dists_ub, h1_dists_ub = return_mmd_distributions(
    h0_paths, h1_paths, signature_kernel.compute_mmd,
    n_atoms=n_atoms, batch_size=n_paths, estimator="ub",
)
plot_dist(h0_dists_ub, h1_dists_ub, n_atoms, alpha, "mmd_poisson_unbiased.svg", svg=True)
plt.show()

# -- Biased --
print("Computing biased MMD distributions...")
h0_dists_b, h1_dists_b = return_mmd_distributions(
    h0_paths, h1_paths, signature_kernel.compute_mmd,
    n_atoms=n_atoms, batch_size=n_paths, estimator="b",
)
plot_dist(h0_dists_b, h1_dists_b, n_atoms, alpha, "mmd_poisson_biased.svg", svg=True)
plt.show()

# ---------------------------------------------------------------------------
# 6. Level contributions
# ---------------------------------------------------------------------------
ks = [1, 2, 3, 4]
n_atoms_lvl = 2048
n_paths_lvl = 128

print("Computing level contributions (unbiased)...")
h0_Mk, h1_Mk = get_level_values(h0_paths, h1_paths, n_atoms_lvl, n_paths_lvl, ks, path_bank_size)
h0_Mk = np.asarray(h0_Mk)
h1_Mk = np.asarray(h1_Mk)
plot_level_contributions(h0_Mk, h1_Mk, n_atoms_lvl, ks, "mmd_level_terms_poisson_unbiased.svg",
                         svg=True, scientific=True, filter=False)
plt.show()

# ---------------------------------------------------------------------------
# 7. Apply scaling
# ---------------------------------------------------------------------------
scaling = 5.5


def filter_nan(dists):
    """Remove NaN and infinite values from a list of MMD estimates."""
    clean = [d for d in dists if np.isfinite(d)]
    n_removed = len(dists) - len(clean)
    if n_removed > 0:
        print(f"  Warning: removed {n_removed}/{len(dists)} non-finite MMD values.")
    return clean


print("Computing scaled MMD distributions (unbiased)...")
h0_sc_ub, h1_sc_ub = return_mmd_distributions(
    torch.multiply(torch.Tensor([scaling, 1]).to(device=device), h0_paths),
    torch.multiply(torch.Tensor([scaling, 1]).to(device=device), h1_paths),
    signature_kernel.compute_mmd,
    n_atoms=n_atoms, batch_size=n_paths, estimator="ub",
)
h0_sc_ub, h1_sc_ub = filter_nan(h0_sc_ub), filter_nan(h1_sc_ub)
if h0_sc_ub and h1_sc_ub:
    plot_dist(h0_sc_ub, h1_sc_ub, len(h0_sc_ub), alpha, f"mmd_poisson_scaling_{scaling}.svg", svg=True)
    plt.show()
else:
    print("  Skipping plot: all MMD values are non-finite for this scaling.")

print("Computing scaled level contributions...")
h0_sc_Mk, h1_sc_Mk = get_level_values(
    torch.multiply(torch.Tensor([scaling, 1]).to(device=device), h0_paths),
    torch.multiply(torch.Tensor([scaling, 1]).to(device=device), h1_paths),
    n_atoms_lvl, n_paths_lvl, ks, path_bank_size,
)
h0_sc_Mk = np.asarray(h0_sc_Mk)
h1_sc_Mk = np.asarray(h1_sc_Mk)
plot_level_contributions(h0_sc_Mk, h1_sc_Mk, n_atoms_lvl, ks,
                         f"mmd_level_terms_poisson_scaling_{scaling}.svg", filter=False, svg=True)
plt.show()

print("Computing scaled MMD distributions (biased)...")
h0_sc_b, h1_sc_b = return_mmd_distributions(
    torch.multiply(torch.Tensor([scaling, 1]).to(device=device), h0_paths),
    torch.multiply(torch.Tensor([scaling, 1]).to(device=device), h1_paths),
    signature_kernel.compute_mmd,
    n_atoms=n_atoms, batch_size=n_paths, estimator="b",
)
h0_sc_b, h1_sc_b = filter_nan(h0_sc_b), filter_nan(h1_sc_b)
if h0_sc_b and h1_sc_b:
    plot_dist(h0_sc_b, h1_sc_b, len(h0_sc_b), alpha, f"mmd_poisson_scaling_{scaling}_biased.svg", svg=True)
    plt.show()
else:
    print("  Skipping plot: all MMD values are non-finite for this scaling.")

# ---------------------------------------------------------------------------
# 8. Analysis: Errors vs Lambda Difference
#    For each delta_lam = lam_1 - lam_0, simulate fresh paths and compute
#    the empirical Type 1 and Type 2 error probabilities.
# ---------------------------------------------------------------------------

def compute_errors_for_lambda(
    lam_0, delta_lams, path_bank_size, grid_points, T,
    n_atoms, n_paths, alpha, estimator, num_rep, fixed_scaling=1.0
):
    """
    For each delta_lam in delta_lams, simulate H0 (lam_0) and H1 (lam_0 + delta_lam)
    path banks and compute Type 1 / Type 2 error probabilities.

    Repeats `num_rep` times to build confidence bands.

    Returns
    -------
    type1_results : dict  {delta_lam -> list of type1_error over num_rep}
    type2_results : dict  {delta_lam -> list of type2_error over num_rep}
    """
    type1_results = defaultdict(list)
    type2_results = defaultdict(list)

    for rep in tqdm(range(num_rep), desc="Lambda analysis repetitions"):
        for dl in delta_lams:
            lam_1_cur = lam_0 + dl
            h0, h1 = load_paths(lam_0, lam_1_cur, path_bank_size, grid_points, T)
            t2e, t1e = get_type1_type2_errors(
                signature_kernel, h0, h1,
                scaling=fixed_scaling,
                n_atoms=n_atoms,
                n_paths=n_paths,
                estimator=estimator,
                alpha=alpha,
                device=device,
            )
            type1_results[dl].append(100.0 - float(t1e))  # get_type1_type2_errors returns P[not reject|H0], so subtract from 100
            type2_results[dl].append(float(t2e))

    return type1_results, type2_results


def plot_errors_vs_lambda(type1_results, type2_results, delta_lams, title_suffix="", filename_prefix="poisson_lambda"):
    """
    Plot Type 1 and Type 2 errors (mean ± std) as a function of delta_lam.
    """
    dl_arr = np.array(delta_lams)

    t1_mean = np.array([np.mean(type1_results[dl]) for dl in delta_lams])
    t1_std  = np.array([np.std(type1_results[dl])  for dl in delta_lams])

    t2_mean = np.array([np.mean(type2_results[dl]) for dl in delta_lams])
    t2_std  = np.array([np.std(type2_results[dl])  for dl in delta_lams])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mean, std, label, color in zip(
        axes,
        [t1_mean, t2_mean],
        [t1_std, t2_std],
        ["Type 1 Error (%)", "Type 2 Error (%)"],
        ["tomato", "dodgerblue"],
    ):
        ax.plot(dl_arr, mean, color=color, linewidth=2)
        ax.fill_between(dl_arr, mean - std, mean + std, color=color, alpha=0.3, edgecolor="none")
        ax.set_xlabel(r"$\lambda_1 - \lambda_0$", fontsize=14)
        ax.set_ylabel(label, fontsize=14)
        ax.set_title(f"{label} vs $\Delta\lambda$ — {title_suffix}", fontsize=13)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True, color="black", alpha=0.15)
        plt.setp(ax.get_xticklabels(), fontsize=12)
        plt.setp(ax.get_yticklabels(), fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_errors.svg", bbox_inches="tight", format="svg")
    plt.show()


# Parameters for lambda analysis
delta_lams   = np.linspace(0, 10, 15)   # delta = lam_1 - lam_0
n_atoms_lam  = 200
n_paths_lam  = 64
num_rep_lam  = 20
fixed_scaling = 1.0

print("\n--- Analysis: Errors vs Lambda Difference (unbiased) ---")
t1_lam_ub, t2_lam_ub = compute_errors_for_lambda(
    lam_0, delta_lams, path_bank_size, grid_points, T,
    n_atoms_lam, n_paths_lam, alpha, "ub", num_rep_lam, fixed_scaling,
)
plot_errors_vs_lambda(t1_lam_ub, t2_lam_ub, delta_lams, title_suffix="Unbiased",
                      filename_prefix="poisson_lambda_unbiased")

print("\n--- Analysis: Errors vs Lambda Difference (biased) ---")
t1_lam_b, t2_lam_b = compute_errors_for_lambda(
    lam_0, delta_lams, path_bank_size, grid_points, T,
    n_atoms_lam, n_paths_lam, alpha, "b", num_rep_lam, fixed_scaling,
)
plot_errors_vs_lambda(t1_lam_b, t2_lam_b, delta_lams, title_suffix="Biased",
                      filename_prefix="poisson_lambda_biased")

# ---------------------------------------------------------------------------
# 9. Analysis: Errors vs Scaling and Number of Examples
# ---------------------------------------------------------------------------
scalings       = np.linspace(0, 5, 20)
n_atoms_err    = 100
n_paths_list   = [20, 40, 60, 120]
num_sim        = 100
alpha_err      = 0.05

os.makedirs("PoissonData", exist_ok=True)

print("\n--- Analysis: Errors vs Scaling & Batch Size (unbiased) ---")
type1_ub, type2_ub = generate_error_probs_linear_kernel(
    signature_kernel, h0_paths, h1_paths,
    n_atoms_err, n_paths_list, alpha_err, scalings,
    "ub", num_sim, device,
    filename="poisson_unbiased", folder="PoissonData/",
)

print("\n--- Analysis: Errors vs Scaling & Batch Size (biased) ---")
type1_b, type2_b = generate_error_probs_linear_kernel(
    signature_kernel, h0_paths, h1_paths,
    n_atoms_err, n_paths_list, alpha_err, scalings,
    "b", num_sim, device,
    filename="poisson_biased", folder="PoissonData/",
)

# -- Or reload saved results --
# with open("PoissonData/type1error_poisson_unbiased", "rb") as fp: type1_ub = pickle.load(fp)
# with open("PoissonData/type2error_poisson_unbiased", "rb") as fp: type2_ub = pickle.load(fp)
# with open("PoissonData/type1error_poisson_biased",   "rb") as fp: type1_b  = pickle.load(fp)
# with open("PoissonData/type2error_poisson_biased",   "rb") as fp: type2_b  = pickle.load(fp)

plot_type2_error(type2_ub, scalings, n_paths_list, title="Unbiased — Type 2 Error vs Scaling")
plt.savefig("poisson_type2_unbiased.svg", bbox_inches="tight", format="svg")
plt.show()

plot_type2_error(type2_b, scalings, n_paths_list, title="Biased — Type 2 Error vs Scaling")
plt.savefig("poisson_type2_biased.svg", bbox_inches="tight", format="svg")
plt.show()

plot_type1_error(type1_ub, scalings, n_paths_list, title="Unbiased — Type 1 Error vs Scaling")
plt.savefig("poisson_type1_unbiased.svg", bbox_inches="tight", format="svg")
plt.show()

plot_type1_error(type1_b, scalings, n_paths_list, title="Biased — Type 1 Error vs Scaling")
plt.savefig("poisson_type1_biased.svg", bbox_inches="tight", format="svg")
plt.show()
