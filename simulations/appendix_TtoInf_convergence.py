#!/usr/bin/env python3
"""
Generate the finite-horizon consistency check plot (Figure S1) from
appendix_TtoInf_convergence.npz.

This script DOES NOT recompute simulations. It only visualizes:
- kappa*(T) = argmax_kappa P_success(T) for fixed gamma slices
- kappa*_g = argmax_kappa g(gamma,kappa) as horizontal dashed line

Expected input keys in the .npz (as provided in this repository):
T_list, kappas, results (dict keyed by gamma), N, eta, seed, V_pot

The stored 'results' dict contains for each gamma:
- kappa_opt_g (float)
- kappa_opt_P (array over T_list)
- gaps (array over kappas)
- P_T (matrix [len(T_list), len(kappas)])
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _repo_root() -> Path:
    # simulations/appendix_TtoInf_convergence.py -> repo root is parent of simulations/
    return Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        type=str,
        default=str(_repo_root() / "data" / "appendix_TtoInf_convergence.npz"),
        help="Path to appendix_TtoInf_convergence.npz",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_repo_root() / "appendix"),
        help="Output directory for rendered appendix figures",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG DPI",
    )
    args = ap.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    T_list = np.array(data["T_list"], dtype=float)
    results = data["results"].item()

    gammas = sorted(results.keys())

    fig, axs = plt.subplots(1, len(gammas), figsize=(12, 4.2), sharey=True, constrained_layout=True)
    if len(gammas) == 1:
        axs = [axs]

    for ax, g in zip(axs, gammas):
        r = results[g]
        kappa_opt_g = float(r["kappa_opt_g"])
        kappa_opt_P = np.array(r["kappa_opt_P"], dtype=float)

        ax.plot(T_list, kappa_opt_P, marker="o")
        ax.axhline(kappa_opt_g, linestyle="--", linewidth=2)

        ax.set_xscale("log")
        ax.set_xlabel(r"Horizon $T$ (log scale)")
        ax.set_title(rf"$\gamma={float(g):.1f}$ slice")
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    axs[0].set_ylabel(r"Optimal dephasing $\kappa^*$")
    # Legend only once
    axs[0].plot([], [], marker="o", linestyle="-", label=r"$\arg\max_\kappa P_{\mathrm{success}}(T)$")
    axs[0].plot([], [], linestyle="--", label=r"$\arg\max_\kappa g$")
    axs[0].legend(loc="upper left", frameon=True)

    try:
        N = int(np.array(data["N"]).item())
        eta = float(np.array(data["eta"]).item())
        seed = int(np.array(data["seed"]).item())
        fig.suptitle(f"Finite-horizon vs spectral optimum (N={N}, eta={eta}, seed={seed})", y=1.03, fontsize=10)
    except Exception:
        pass

    out_png = out_dir / "Figure_S1_TtoInf_convergence.png"
    out_svg = out_dir / "Figure_S1_TtoInf_convergence.svg"
    fig.savefig(out_png, dpi=args.dpi)
    fig.savefig(out_svg)
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
