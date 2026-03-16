import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

p1 = "eq_hawkes_fixed_mean/linear/detailed_scenario/data/type1error_hawkes_fixed_mean_worst_case_alpha_4.0"
p2 = "eq_hawkes_fixed_mean/linear/detailed_scenario/data/type2error_hawkes_fixed_mean_worst_case_alpha_4.0"

with open(p1, "rb") as f: type1_list = pickle.load(f)
with open(p2, "rb") as f: type2_list = pickle.load(f)

# On va supprimer la DERNIERE valeur (qui correspond au point 'scaling=5') qui a dégénéré (à 0.)
# On a aussi accidentellement supprimé l'indice 5 précédemment (la longueur est de 39 au lieu de 40).
# On ne gardera donc que les 38 premieres valeurs pour que ça matche.

for sim_dict in type1_list:
    for n_paths, errors in sim_dict.items():
        if len(errors) == 39:
            del errors[-1] # delete last one

for sim_dict in type2_list:
    for n_paths, errors in sim_dict.items():
        if len(errors) == 39:
            del errors[-1]

# Recreate an appropriate scalings array of length 38!
# If original was np.linspace(0, 5, 40), it had 40 points.
orig_scalings = np.linspace(0, 5, 40)
# we remove index 5
scalings = np.delete(orig_scalings, 5)
# we remove the last one (index 38)
scalings = np.delete(scalings, -1)

n_paths_err_list = [20, 60, 120]
num_sim = 25

def plot_type2_error_n(type2_list, scalings, n_paths_list, num_sim, title="", filename=None, colors=None):
    if colors is None: colors = ["magenta", "green", "darkorange", "blue"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, n_paths in enumerate(n_paths_list):
        t2e = [type2_list[j][n_paths] for j in range(num_sim)]
        t2e_mean = np.mean(np.asarray(t2e), axis=0)
        t2e_std  = np.std(np.asarray(t2e), axis=0)
        ax.plot(scalings, t2e_mean, alpha=1, label=f"{n_paths}", color=colors[i])
        ax.fill_between(scalings, t2e_mean - t2e_std, t2e_mean + t2e_std, color=colors[i], alpha=0.3, edgecolor="none")
    ax.set_ylabel("P[Type 2 Error] (%)", fontsize=13)
    ax.set_xlabel("Scaling", fontsize=13)
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=13)
    ax.grid(True, color="black", alpha=0.2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.title(title, fontsize=13)
    if filename: plt.savefig(filename, bbox_inches="tight", format="svg")
    plt.close()

def plot_type1_error_n(type1_list, scalings, n_paths_list, num_sim, title="", filename=None, colors=None):
    if colors is None: colors = ["magenta", "green", "darkorange", "blue"]
    n = len(n_paths_list)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for i, n_paths in enumerate(n_paths_list):
        ax = axes_flat[i]
        t1e = [100 - np.asarray(type1_list[j][n_paths]) for j in range(num_sim)]
        bp = ax.boxplot(np.asarray(t1e)[:, 1::2], patch_artist=True, labels=np.round(scalings[1::2], 2))
        for patch in bp["boxes"]: patch.set_facecolor(colors[i % len(colors)]); patch.set_alpha(0.3)
        for median in bp["medians"]: median.set(color=colors[i % len(colors)], linewidth=3)
        for whisker in bp["whiskers"]: whisker.set(color=colors[i % len(colors)], linewidth=2.5, linestyle=":")
        for cap in bp["caps"]: cap.set(color=colors[i % len(colors)], linewidth=3)
        for flier in bp["fliers"]: flier.set(markeredgecolor=colors[i % len(colors)], markerfacecolor=colors[i % len(colors)], alpha=0.75)
        ax.set_xlabel("Scaling", fontsize=12)
        ax.set_ylabel("P[Type 1 Error] (%)", fontsize=12)
        ax.set_title(f"Batch Size: {n_paths}", fontsize=12)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    for j in range(len(n_paths_list), len(axes_flat)): axes_flat[j].set_visible(False)
    fig.suptitle(title, fontsize=14, y=1.0)
    plt.subplots_adjust(hspace=0.3)
    if filename: plt.savefig(filename, bbox_inches="tight", format="svg")
    plt.close()

plot_type2_error_n(type2_list, scalings, n_paths_err_list, num_sim,
                   title="fixed_mean_worst_case_alpha_4.0 — Type 2 Error",
                   filename="eq_hawkes_fixed_mean/linear/detailed_scenario/type2_fixed_mean_worst_case_alpha_4.0.svg")

plot_type1_error_n(type1_list, scalings, n_paths_err_list, num_sim,
                   title="fixed_mean_worst_case_alpha_4.0 — Type 1 Error",
                   filename="eq_hawkes_fixed_mean/linear/detailed_scenario/type1_fixed_mean_worst_case_alpha_4.0.svg")

print("Plots successfully regenerated with extreme point (scaling=5.0) removed!")
