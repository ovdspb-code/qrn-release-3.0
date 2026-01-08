# Quantum-Resonant Netting (QRN) — Release 3.0

This repository contains the numerical simulation code and associated data
used in the QRN release 3.0 manuscript:

“Quantum-Resonant Netting (QRN): Thermodynamic motivation, formal model of open
wave dynamics on the cognitome, and validation program”.

The purpose of this repository is transparency and reproducibility of the
numerical illustrations presented in Sections 5 and Appendices D–E of the paper.

---

## Repository structure

simulations/
    Python scripts used to generate the numerical results and figures.
    The code is provided as-is and reflects the exact version used for the
    release 3.0 manuscript.

data/
    Precomputed numerical data (NumPy .npz files) produced by the simulations
    and used to generate figures.

figures/
    Rendered figures corresponding to Figures 2–5 of the main text.
    These are provided for reference and visual inspection.

appendix/
    Supplementary figures corresponding to Appendix D (Figures S1 and S2).

---

## Mapping to the manuscript

Main text:
- Figure 2: Targeting dynamics (P_success vs time)
- Figure 3: Expected potential ⟨V⟩(t)
- Figure 4: Efficiency landscape P_success(T; γ, κ)
- Figure 5: Liouvillian spectral gap g(γ, κ)

Appendices:
- Appendix D, Figure S1: Finite-horizon vs spectral optimum convergence
- Appendix D, Figure S2: Robustness of κ-optimum vs readout rate η
- Appendix E: Operational checkpoints and payoff maps

---

## Reproducibility notes

- The simulations are intentionally small-scale (N = 10) and serve as a
  qualitative illustration of the ENAQT-like regime discussed in the paper.
- Full density-matrix simulations scale poorly with N; the numerical examples
  are not intended as large-scale brain models.
- No claim is made that the code represents a biologically realistic substrate.

---

## License

MIT License. See LICENSE file for details.

---

## Citation

If you use or refer to this code, please cite the QRN release 3.0 manuscript.