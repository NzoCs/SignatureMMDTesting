import os
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")

# Importation depuis vos fichiers existants
from hawkes_fixed_mean_analysis import (
    load_hawkes_paths,
    plot_type1_error_n,
    plot_type2_error_n,
)
from src.mmd.mmd import SigKernel, LinearKernel
from src.mmd.distribution_functions import generate_error_probs_linear_kernel


def main():
    # Setup Device & Kernel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dyadic_order = 0
    sig_kernel = SigKernel(static_kernel=LinearKernel(), dyadic_order=dyadic_order)

    # Paramètres d'expérience
    DATA_DIR = "eq_hawkes_fixed_mean/linear/detailed_scenario_alpha4"
    os.makedirs(f"{DATA_DIR}/data", exist_ok=True)

    T = 1
    grid_points = 100
    fixed_beta = 10.0

    # Paramètres de H0
    alpha0 = 5.0
    mu0 = 100.0 - 10.0 * alpha0  # 50.0

    # Paramètres de H1 ciblés (alpha = 4.0)
    alpha_extreme = 4.0
    mu_extreme = 100.0 - 10.0 * alpha_extreme  # 60.0

    # Paramètres d'évaluation
    path_bank_size = 10000
    alpha_test = 0.05
    scalings_sweep = np.linspace(0.5, 4.9, 20)

    # Tailles de batch à évaluer
    n_paths_err_list = [240, 480]
    n_atoms_err = 50
    num_sim = 25

    print(f"Génération des chemins...")
    print(f"H0: mu={mu0}, alpha={alpha0}, beta={fixed_beta}")
    print(f"H1: mu={mu_extreme}, alpha={alpha_extreme}, beta={fixed_beta}")

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

    label = f"alpha_4.0_only"
    print(f"Calcul des erreurs Type I et Type II vs Scaling pour alpha = 4.0...")

    type1_list, type2_list = generate_error_probs_linear_kernel(
        sig_kernel,
        h0_s,
        h1_s,
        n_atoms_err,
        n_paths_err_list,
        alpha_test,
        scalings_sweep,
        "ub",
        num_sim,
        device,
        filename=f"hawkes_{label}",
        folder=f"{DATA_DIR}/data/",
    )

    print("Génération des graphiques...")
    plot_type2_error_n(
        type2_list,
        scalings_sweep,
        n_paths_err_list,
        num_sim,
        title=f"Alpha=4.0 — Type 2 Error vs Scaling",
        filename=f"{DATA_DIR}/type2_{label}.svg",
    )

    plot_type1_error_n(
        type1_list,
        scalings_sweep,
        n_paths_err_list,
        num_sim,
        title=f"Alpha=4.0 — Type 1 Error vs Scaling",
        filename=f"{DATA_DIR}/type1_{label}.svg",
    )

    print(f"Terminé ! Résultats sauvegardés dans : {DATA_DIR}")


if __name__ == "__main__":
    main()
