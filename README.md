# Quantum-Resonant Netting (QRN) — Release 3.0

This repository contains the numerical simulation code and numerical data
associated with the manuscript:

“Quantum-Resonant Netting (QRN): Thermodynamic motivation, formal model of open
wave dynamics on the cognitome, and validation program”.

The repository is provided for transparency and reproducibility of the
numerical illustrations presented in the main text (Figures 2–5) and in the
appendices (Figures S1–S2, Appendix D/E) of the release 3.0 manuscript.

---

## Repository structure

simulations/
    Python scripts used to generate figures from precomputed numerical data.
    The scripts do NOT recompute simulations; they only visualize stored
    results.

data/
    Precomputed numerical data in NumPy (.npz) format produced by the QRN toy
    simulations.

figures/
    Rendered figures corresponding to Figures 2–5 of the main text.

appendix/
    Rendered supplementary figures corresponding to Appendix D
    (Figures S1 and S2).

---

## Mapping to the manuscript

Main text:
- Figure 2: Targeting dynamics (P_success vs time)
- Figure 3: Expected potential ⟨V⟩(t)
- Figure 4: Efficiency landscape P_success(T; γ, κ)
- Figure 5: P_success(T) and Liouvillian spectral gap g(γ, κ)

Appendices:
- Appendix D, Figure S1: Finite-horizon convergence of κ*
- Appendix D, Figure S2: Robustness of κ* versus readout/sink rate η
- Appendix E: Operational and payoff checkpoints

---

## How to reproduce the figures

All scripts operate on precomputed data in the `data/` directory and do not
perform new simulations.

Figure 2–4 (main text):

python simulations/qrn_generate_fig2_4.py


Figure 5 (main text):

python simulations/qrn_generate_fig5_gap.py


Figure S1 (appendix):

python simulations/appendix_TtoInf_convergence.py


Generated figures are saved into the `figures/` or `appendix/` directories.

---

## Reproducibility notes

- The simulations correspond to a minimal toy model (N = 10) intended to
  demonstrate the qualitative ENAQT-like regime discussed in the paper.
- Full density-matrix simulations scale poorly with system size and are not
  intended as large-scale brain models.
- The code is provided exactly as used for the release 3.0 manuscript, without
  optimization or refactoring.

---

## License

This project is released under the MIT License. See the LICENSE file for
details.

---

## Citation

If you use or refer to this repository, please cite the QRN release 3.0
manuscript.

