#!/usr/bin/env python3
"""
Generate Figure 5 from precomputed maps_Psuccess_gap.npz.

This script DOES NOT recompute simulations. It only visualizes:
(A) P_success(T; gamma, kappa)
(B) Liouvillian spectral gap g(gamma, kappa)

It overlays the ridge lines:
- dashed: kappa*(gamma) = argmax_kappa P_success(T; gamma, kappa)
- dotted: kappa*_g(gamma) = argmax_kappa g(gamma, kappa)

Expected input keys in the .npz (as provided in this repository):
gammas, kappas, P_map, G_map, N, eta, T, seed, V_pot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _repo_root() -> Path:
    # simulations/qrn_generate_fig5_gap.py -> repo root is parent of simulations/
    return Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz",
        type=str,
        default=str(_repo_root() / "data" / "maps_Psuccess_gap.npz"),
        help="Path to maps_Psuccess_gap.npz",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_repo_root() / "figures"),
        help="Output directory for rendered figures",
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
    gammas = np.array(data["gammas"], dtype=float)
    kappas = np.array(data["kappas"], dtype=float)
    P = np.array(data["P_map"], dtype=float)
    G = np.array(data["G_map"], dtype=float)

    if P.shape != (len(kappas), len(gammas)):
        raise ValueError(
            f"Unexpected P_map shape {P.shape}; expected ({len(kappas)}, {len(gammas)})"
        )
    if G.shape != (len(kappas), len(gammas)):
        raise ValueError(
            f"Unexpected G_map shape {G.shape}; expected ({len(kappas)}, {len(gammas)})"
        )

    # ridge lines across gamma
    kappa_star_P = kappas[np.argmax(P, axis=0)]  # per gamma
    kappa_star_G = kappas[np.argmax(G, axis=0)]

    # global maxima
    idxP = np.unravel_index(np.argmax(P), P.shape)
    idxG = np.unravel_index(np.argmax(G), G.shape)
    gP, kP, maxP = gammas[idxP[1]], kappas[idxP[0]], P[idxP]
    gG, kG, maxG = gammas[idxG[1]], kappas[idxG[0]], G[idxG]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    extent = [gammas.min(), gammas.max(), kappas.min(), kappas.max()]

    im0 = axs[0].imshow(P, origin="lower", aspect="auto", extent=extent)
    axs[0].plot(gammas, kappa_star_P, linestyle="--", linewidth=2, label=r"$\kappa^*(\gamma)$")
    axs[0].plot(gammas, kappa_star_G, linestyle=":", linewidth=2, label=r"$\kappa_g^*(\gamma)$")
    axs[0].scatter([gP], [kP], s=60, marker="o")
    axs[0].set_xlabel(r"$\gamma$")
    axs[0].set_ylabel(r"$\kappa$")
    axs[0].set_title(r"(A) $P_{\mathrm{success}}(T;\gamma,\kappa)$")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(G, origin="lower", aspect="auto", extent=extent)
    axs[1].plot(gammas, kappa_star_P, linestyle="--", linewidth=2, label=r"$\kappa^*(\gamma)$")
    axs[1].plot(gammas, kappa_star_G, linestyle=":", linewidth=2, label=r"$\kappa_g^*(\gamma)$")
    axs[1].scatter([gG], [kG], s=60, marker="o")
    axs[1].set_xlabel(r"$\gamma$")
    axs[1].set_ylabel(r"$\kappa$")
    axs[1].set_title(r"(B) $g(\gamma,\kappa)$")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    axs[1].legend(loc="upper left", frameon=True)

    # Minimal provenance line (optional)
    try:
        N = int(np.array(data["N"]).item())
        eta = float(np.array(data["eta"]).item())
        T = float(np.array(data["T"]).item())
        seed = int(np.array(data["seed"]).item())
        fig.suptitle(f"QRN toy model maps (N={N}, eta={eta}, T={T}, seed={seed})", y=1.02, fontsize=10)
    except Exception:
        pass

    out_png = out_dir / "Fig5_QRN_Psuccess_vs_LiouvillianGap.png"
    out_svg = out_dir / "Fig5_QRN_Psuccess_vs_LiouvillianGap.svg"
    fig.savefig(out_png, dpi=args.dpi)
    fig.savefig(out_svg)
    plt.close(fig)

    print("Saved:", out_png)
    print("Saved:", out_svg)
    print(f"Global max P_success: gamma={gP:.3g}, kappa={kP:.3g}, P={maxP:.6g}")
    print(f"Global max gap g:     gamma={gG:.3g}, kappa={kG:.3g}, g={maxG:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
