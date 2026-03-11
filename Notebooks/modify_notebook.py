import json
import os

notebook_path = r"c:\Users\enzo.cAo\Documents\Projects\projet_recherche\SignatureMMDTesting\Notebooks\Poisson Processes.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 0: Title
nb["cells"][0]["source"] = [
    "# Signature MMD Two-Sample Statistical Tests\n",
    "\n",
    "## Numerical Examples - Poisson Processes\n",
    "\n",
    "### Andrew Alden, Blanka Horvath, Zacharia Issa",
]

# Find the cell that contains `def sim_garch_model`
garch_def_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and any(
        "def sim_garch_model" in line for line in cell["source"]
    ):
        garch_def_idx = i
        break

if garch_def_idx != -1:
    nb["cells"][garch_def_idx]["source"] = [
        "def sim_poisson_model(lam, num_sim, num_time_steps, T):\n",
        "    time_steps = np.linspace(0, T, num_time_steps)\n",
        "    dt = T / (num_time_steps - 1)\n",
        "    # Increments\n",
        "    increments = np.random.poisson(lam * dt, size=(num_time_steps - 1, num_sim))\n",
        "    # Paths (num_time_steps, num_sim)\n",
        "    paths = np.vstack([np.zeros((1, num_sim)), np.cumsum(increments, axis=0)])\n",
        "    # Combine with time steps\n",
        "    # path_bank shape: (num_time_steps, num_sim, 2)\n",
        "    return np.concatenate((paths[:, :, None], np.repeat(np.asarray(time_steps)[:, None, None], repeats=num_sim, axis=1)), axis=2)",
    ]

# Find the cell that contains `sim1 = arch.arch_model`
arch_model_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and any(
        "arch.arch_model" in line for line in cell["source"]
    ):
        arch_model_idx = i
        break

if arch_model_idx != -1:
    nb["cells"][arch_model_idx]["source"] = [
        "# H0 intensity\n",
        "lam_0 = 10.0\n",
        "# H1 intensity\n",
        "lam_1 = 12.0",
    ]

# Find the cell that contains `h0_path_bank = sim_garch_model`
path_bank_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and any(
        "h0_path_bank = sim_garch_model" in line for line in cell["source"]
    ):
        path_bank_idx = i
        break

if path_bank_idx != -1:
    nb["cells"][path_bank_idx]["source"] = [
        "T = 1\n",
        "grid_points = 20\n",
        "path_bank_size = 10000\n",
        "\n",
        "h0_path_bank = sim_poisson_model(lam_0, path_bank_size, grid_points, T)\n",
        "h1_path_bank = sim_poisson_model(lam_1, path_bank_size, grid_points, T)\n",
        "\n",
        "h0_paths = torch.transpose(torch.from_numpy(h0_path_bank), 0, 1).to(device=device, dtype=torch.float32)\n",
        "h1_paths = torch.transpose(torch.from_numpy(h1_path_bank), 0, 1).to(device=device, dtype=torch.float32)\n",
        "\n",
        "for i in range(path_bank_size):\n",
        "    h0_paths[i, :, :] = h0_paths[i, :, :] - h0_paths[i, 0, :]\n",
        "    h1_paths[i, :, :] = h1_paths[i, :, :] - h1_paths[i, 0, :]",
    ]

# Replace '_garch' with '_poisson' in the rest of the file
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        for i in range(len(cell["source"])):
            cell["source"][i] = cell["source"][i].replace("_garch", "_poisson")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("done")
