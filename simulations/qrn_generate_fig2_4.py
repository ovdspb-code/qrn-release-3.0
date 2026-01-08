#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate QRN toy-model figures (Fig2–Fig4 + optional Fig4b gap heatmap).

Outputs (to --out-dir):
- Fig2_QRN_Targeting_300dpi.png / .svg
- Fig3_QRN_ExpectedPotential_300dpi.png / .svg
- Fig4_QRN_EfficiencyLandscape_300dpi.png / .svg
- (optional) Fig4b_QRN_LiouvillianGapLandscape_300dpi.png / .svg

Speed:
- Figure 4 heatmap supports parallel execution (threads/processes), chunking, progress+ETA.
- Optional gap heatmap supports parallel execution; eigen-solve can be dense or sparse (ARPACK).

Notes:
- Uses an explicit sink basis state so Tr(rho)=1 and success probability is sink population.
- Vectorization convention: vec_F(A) stacks columns (Fortran order).
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Literal, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # why: allow running on headless machines/CI
import matplotlib.pyplot as plt

import scipy.linalg as la
from scipy.sparse import csr_matrix, identity, kron as sp_kron
from scipy.sparse.linalg import eigs, expm_multiply

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


# ======================
# Defaults (Table-like)
# ======================
DEFAULT_N = 10
DEFAULT_T_HORIZON = 20.0
DEFAULT_GAMMA = 2.0
DEFAULT_ETA = 0.8


# =========================
# Linear-algebra utilities
# =========================

def laplacian_chain(n: int) -> np.ndarray:
    """Graph Laplacian L = D - A for an undirected chain of n nodes."""
    if n < 2:
        raise ValueError("n must be >= 2")
    A = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    D = np.diag(A.sum(axis=1))
    return D - A


def vecF(M: np.ndarray) -> np.ndarray:
    """Column-stacking vec operator (Fortran order)."""
    return np.asarray(M, dtype=complex).reshape((-1,), order="F")


def matF(v: np.ndarray, dim: int) -> np.ndarray:
    """Inverse vecF: reshape vector back into dim x dim matrix (Fortran order)."""
    return np.asarray(v, dtype=complex).reshape((dim, dim), order="F")


def embed_system_operator(op_sys: np.ndarray, dim_sink: int = 1) -> np.ndarray:
    """Embed an NxN operator on the system into (N+dim_sink)x(N+dim_sink) with zeros on sink block."""
    n = op_sys.shape[0]
    d = n + dim_sink
    out = np.zeros((d, d), dtype=complex)
    out[:n, :n] = op_sys.astype(complex)
    return out


def projector(dim: int, i: int) -> np.ndarray:
    """|i><i|."""
    P = np.zeros((dim, dim), dtype=complex)
    P[i, i] = 1.0 + 0j
    return P


def commutator_superop_dense(H: np.ndarray) -> np.ndarray:
    """Superoperator for -i[H, .] using vecF convention (dense)."""
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    return -1j * (np.kron(I, H) - np.kron(H.T, I))


def dissipator_superop_dense(L: np.ndarray) -> np.ndarray:
    """Superoperator D_L(.) = L . L† - 1/2 {L†L, .} using vecF convention (dense)."""
    d = L.shape[0]
    I = np.eye(d, dtype=complex)
    Ldag = L.conj().T
    LdagL = Ldag @ L
    return np.kron(L, L.conj()) - 0.5 * (np.kron(I, LdagL) + np.kron(LdagL.T, I))


def commutator_superop_sparse(H: csr_matrix) -> csr_matrix:
    """Sparse superoperator for -i[H, .] using vecF convention."""
    d = H.shape[0]
    I = identity(d, dtype=complex, format="csr")
    return (-1j * (sp_kron(I, H, format="csr") - sp_kron(H.T, I, format="csr"))).tocsr()


def dissipator_superop_sparse(L: csr_matrix) -> csr_matrix:
    """Sparse superoperator D_L(.) using vecF convention."""
    d = L.shape[0]
    I = identity(d, dtype=complex, format="csr")
    Ldag = L.conjugate().transpose()
    LdagL = (Ldag @ L).tocsr()
    return (sp_kron(L, L.conjugate(), format="csr") - 0.5 * (sp_kron(I, LdagL, format="csr") + sp_kron(LdagL.T, I, format="csr"))).tocsr()


def precompute_liouvillian_components(
    n: int, V_pot_sys: np.ndarray, target: int, *, sparse: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute Liouvillian pieces so L(gamma, kappa) = C_V + eta D_sink + gamma C_L + kappa D_deph.

    Returns (C_L, C_V, D_deph, D_sink) in dense or sparse form.
    """
    d = n + 1  # include sink
    sink = n

    L_graph = laplacian_chain(n)
    H_L = embed_system_operator(-L_graph, dim_sink=1)  # H part that multiplies gamma

    V_sys = np.diag(V_pot_sys.astype(float))
    H_V = embed_system_operator(V_sys, dim_sink=1)  # constant potential part

    if not sparse:
        C_L = commutator_superop_dense(H_L)
        C_V = commutator_superop_dense(H_V)

        D_deph = np.zeros((d * d, d * d), dtype=complex)
        for i in range(n):
            D_deph += dissipator_superop_dense(projector(d, i))

        J = np.zeros((d, d), dtype=complex)
        J[sink, target] = 1.0 + 0j
        D_sink = dissipator_superop_dense(J)

        return C_L, C_V, D_deph, D_sink

    H_Ls = csr_matrix(H_L)
    H_Vs = csr_matrix(H_V)

    C_Ls = commutator_superop_sparse(H_Ls)
    C_Vs = commutator_superop_sparse(H_Vs)

    D_deph_s = csr_matrix((d * d, d * d), dtype=complex)
    for i in range(n):
        Pi = csr_matrix(projector(d, i))
        D_deph_s = (D_deph_s + dissipator_superop_sparse(Pi)).tocsr()

    J = np.zeros((d, d), dtype=complex)
    J[sink, target] = 1.0 + 0j
    D_sink_s = dissipator_superop_sparse(csr_matrix(J))

    return C_Ls, C_Vs, D_deph_s, D_sink_s


# ==========================
# Gap computation utilities
# ==========================

def liouvillian_spectral_gap_from_eigs(evals: np.ndarray, *, tol: float = 1e-10) -> float:
    """Return g = -max Re(lambda) over eigenvalues with Re(lambda) < -tol (ignoring steady modes)."""
    if evals.size == 0:
        return float("nan")
    reals = np.real(evals)
    mask = reals < -abs(tol)
    if not np.any(mask):
        return 0.0
    return float(-np.max(reals[mask]))


def compute_liouvillian_gap(
    N: int,
    gamma: float,
    kappa: float,
    eta: float,
    V_pot: np.ndarray,
    *,
    method: Literal["dense", "sparse"] = "dense",
    k: int = 12,
    tol: float = 1e-10,
) -> float:
    """Compute Liouvillian spectral gap g = -Re(lambda_1) for the QRN model.

    method:
      - dense: full eigvals on dense matrix (scales poorly, robust for small dim)
      - sparse: ARPACK on sparse matrix, returning k eigenvalues with largest Re
    """
    n = int(N)
    d = n + 1

    if method == "dense":
        C_L, C_V, D_deph, D_sink = precompute_liouvillian_components(n, V_pot, n - 1, sparse=False)
        L_super = C_V + float(eta) * D_sink + float(gamma) * C_L + float(kappa) * D_deph
        evals = la.eigvals(L_super)
        return liouvillian_spectral_gap_from_eigs(evals, tol=tol)

    C_Ls, C_Vs, D_deph_s, D_sink_s = precompute_liouvillian_components(n, V_pot, n - 1, sparse=True)
    Ls = (C_Vs + float(eta) * D_sink_s + float(gamma) * C_Ls + float(kappa) * D_deph_s).tocsr()

    kk = int(max(2, min(k, Ls.shape[0] - 2))) if Ls.shape[0] > 4 else 2
    try:
        evals = eigs(Ls, k=kk, which="LR", return_eigenvectors=False)
        return liouvillian_spectral_gap_from_eigs(np.asarray(evals), tol=tol)
    except Exception:
        # Fallback: dense eigvals
        evals = la.eigvals(Ls.toarray())
        return liouvillian_spectral_gap_from_eigs(evals, tol=tol)


# ======================
# Dynamics propagation
# ======================

def propagate_density_timeseries(
    L_super: np.ndarray, rho0: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Return rho(t) on a grid of times (must be evenly-spaced)."""
    d = rho0.shape[0]
    v0 = vecF(rho0)

    t0 = float(times[0])
    t1 = float(times[-1])
    num = int(len(times))

    if not np.allclose(times, np.linspace(t0, t1, num)):
        raise ValueError("'times' must be an evenly spaced linspace")

    vt = expm_multiply(L_super, v0, start=t0, stop=t1, num=num, endpoint=True)

    rhos = np.empty((num, d, d), dtype=complex)
    for k in range(num):
        rhos[k] = matF(vt[k], d)
    return rhos


def crw_generator_chain_with_sink(n: int, gamma: float, eta: float, target: int) -> np.ndarray:
    """Classical random walk generator Q for chain with absorbing sink."""
    d = n + 1
    sink = n
    L = laplacian_chain(n)

    Q = np.zeros((d, d), dtype=float)
    Q[:n, :n] = -gamma * L
    Q[target, target] -= eta
    Q[sink, target] += eta
    return Q


def propagate_prob_timeseries(Q: np.ndarray, p0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Propagate classical probabilities using expm_multiply for Q."""
    t0 = float(times[0])
    t1 = float(times[-1])
    num = int(len(times))
    if not np.allclose(times, np.linspace(t0, t1, num)):
        raise ValueError("'times' must be an evenly spaced linspace")
    pt = expm_multiply(Q, p0, start=t0, stop=t1, num=num, endpoint=True)
    return np.asarray(pt, dtype=float)


def expected_potential_conditional(rho: np.ndarray, V_diag: np.ndarray) -> float:
    """E[V | not yet readout] = Tr(V rho_sys) / Tr(rho_sys); rho_sys excludes sink population."""
    n = V_diag.shape[0]
    rho_sys = rho[:n, :n]
    p_surv = float(np.real(np.trace(rho_sys)))
    if p_surv <= 1e-12:
        return float("nan")
    return float(np.real(np.trace(V_diag @ rho_sys)) / p_surv)


# ==========================
# Parallel heatmap utilities
# ==========================

class _SimpleProgress:
    def __init__(self, total: int, desc: str) -> None:
        self.total = int(total)
        self.desc = desc
        self.start = time.time()
        self.done = 0
        self._last_print = 0.0

    def update(self, n: int = 1) -> None:
        self.done += int(n)
        now = time.time()
        if self.done >= self.total or (now - self._last_print) >= 1.5:
            self._last_print = now
            elapsed = max(now - self.start, 1e-9)
            rate = self.done / elapsed
            remaining = max(self.total - self.done, 0)
            eta = remaining / rate if rate > 1e-12 else float("inf")
            pct = 100.0 * (self.done / max(self.total, 1))
            eta_s = "∞" if not math.isfinite(eta) else f"{eta:,.1f}s"
            print(f"{self.desc}: {self.done}/{self.total} ({pct:5.1f}%) ETA {eta_s}", end="\r")
        if self.done >= self.total:
            print(" " * 100, end="\r")
            print(f"{self.desc}: done in {time.time() - self.start:,.1f}s")

    def close(self) -> None:
        return


# --- Globals for process workers (avoid pickling big arrays per task) ---
_G: dict[str, object] = {}


def _init_efficiency_worker(
    gammas: np.ndarray,
    kappas: np.ndarray,
    C_L_T: np.ndarray,
    L_base_T: np.ndarray,
    D_deph_T: np.ndarray,
    v0: np.ndarray,
    sink_vec_index: int,
) -> None:
    _G["gammas"] = gammas
    _G["kappas"] = kappas
    _G["C_L_T"] = C_L_T
    _G["L_base_T"] = L_base_T
    _G["D_deph_T"] = D_deph_T
    _G["v0"] = v0
    _G["sink_vec_index"] = int(sink_vec_index)


def _efficiency_rows_worker_process(i0: int, i1: int) -> Tuple[int, np.ndarray]:
    gammas = _G["gammas"]  # type: ignore[assignment]
    kappas = _G["kappas"]  # type: ignore[assignment]
    C_L_T = _G["C_L_T"]  # type: ignore[assignment]
    L_base_T = _G["L_base_T"]  # type: ignore[assignment]
    D_deph_T = _G["D_deph_T"]  # type: ignore[assignment]
    v0 = _G["v0"]  # type: ignore[assignment]
    sink_vec_index = int(_G["sink_vec_index"])  # type: ignore[arg-type]

    gammas = np.asarray(gammas, dtype=float)
    kappas = np.asarray(kappas, dtype=float)

    block = np.zeros((i1 - i0, len(gammas)), dtype=float)
    for bi, i in enumerate(range(i0, i1)):
        kappa = float(kappas[i])
        L_k_T = L_base_T + kappa * D_deph_T
        for j, gamma in enumerate(gammas):
            LgkT = L_k_T + float(gamma) * C_L_T
            vT = expm_multiply(LgkT, v0)
            block[bi, j] = float(np.real(vT[sink_vec_index]))
    return i0, block


def _efficiency_rows_worker_thread(
    i0: int,
    i1: int,
    gammas: np.ndarray,
    kappas: np.ndarray,
    C_L_T: np.ndarray,
    L_base_T: np.ndarray,
    D_deph_T: np.ndarray,
    v0: np.ndarray,
    sink_vec_index: int,
) -> Tuple[int, np.ndarray]:
    block = np.zeros((i1 - i0, len(gammas)), dtype=float)
    for bi, i in enumerate(range(i0, i1)):
        kappa = float(kappas[i])
        L_k_T = L_base_T + kappa * D_deph_T
        for j, gamma in enumerate(gammas):
            LgkT = L_k_T + float(gamma) * C_L_T
            vT = expm_multiply(LgkT, v0)
            block[bi, j] = float(np.real(vT[sink_vec_index]))
    return i0, block


def compute_efficiency_heatmap(
    gammas: np.ndarray,
    kappas: np.ndarray,
    C_L: np.ndarray,
    L_base: np.ndarray,
    D_deph: np.ndarray,
    T: float,
    v0: np.ndarray,
    sink_vec_index: int,
    *,
    jobs: int = 1,
    executor: Literal["threads", "processes"] = "threads",
    chunk_rows: int = 0,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute success-probability heatmap at time T."""
    gammas = np.asarray(gammas, dtype=float)
    kappas = np.asarray(kappas, dtype=float)

    nk = int(len(kappas))
    ng = int(len(gammas))
    heatmap = np.zeros((nk, ng), dtype=float)

    jobs = int(max(1, jobs))
    if chunk_rows <= 0:
        target_tasks = max(jobs * 4, 1)
        chunk_rows = max(1, int(math.ceil(nk / target_tasks)))

    tasks: list[tuple[int, int]] = [(i0, min(i0 + chunk_rows, nk)) for i0 in range(0, nk, chunk_rows)]
    total_tasks = len(tasks)

    C_L_T = C_L * float(T)
    L_base_T = L_base * float(T)
    D_deph_T = D_deph * float(T)

    prog = None
    if show_progress:
        prog = tqdm(total=total_tasks, desc="Fig4 heatmap", unit="chunk") if tqdm else _SimpleProgress(total_tasks, "Fig4 heatmap")

    if jobs == 1:
        for i0, i1 in tasks:
            _, block = _efficiency_rows_worker_thread(
                i0, i1, gammas, kappas, C_L_T, L_base_T, D_deph_T, v0, sink_vec_index
            )
            heatmap[i0:i1, :] = block
            if prog:
                prog.update(1)
        if prog and hasattr(prog, "close"):
            prog.close()
        return heatmap

    if executor == "processes":
        try:
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_init_efficiency_worker,
                initargs=(gammas, kappas, C_L_T, L_base_T, D_deph_T, v0, int(sink_vec_index)),
            ) as ex:
                futures = {ex.submit(_efficiency_rows_worker_process, i0, i1): (i0, i1) for (i0, i1) in tasks}
                for fut in as_completed(futures):
                    i0, block = fut.result()
                    heatmap[i0:i0 + block.shape[0], :] = block
                    if prog:
                        prog.update(1)
        except Exception as e:  # pragma: no cover
            print(f"Warning: process executor failed ({type(e).__name__}: {e}); falling back to threads.")
            executor = "threads"

    if executor == "threads":
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {
                ex.submit(
                    _efficiency_rows_worker_thread,
                    i0,
                    i1,
                    gammas,
                    kappas,
                    C_L_T,
                    L_base_T,
                    D_deph_T,
                    v0,
                    int(sink_vec_index),
                ): (i0, i1)
                for (i0, i1) in tasks
            }
            for fut in as_completed(futures):
                i0, block = fut.result()
                heatmap[i0:i0 + block.shape[0], :] = block
                if prog:
                    prog.update(1)

    if prog and hasattr(prog, "close"):
        prog.close()

    return heatmap


# ----- Gap heatmap (parallel) -----

def _init_gap_worker(
    gammas: np.ndarray,
    kappas: np.ndarray,
    C_L: object,
    L_base: object,
    D_deph: object,
    gap_method: str,
    gap_k: int,
    gap_tol: float,
    is_sparse: bool,
) -> None:
    _G["gammas"] = gammas
    _G["kappas"] = kappas
    _G["C_L"] = C_L
    _G["L_base"] = L_base
    _G["D_deph"] = D_deph
    _G["gap_method"] = str(gap_method)
    _G["gap_k"] = int(gap_k)
    _G["gap_tol"] = float(gap_tol)
    _G["is_sparse"] = bool(is_sparse)


def _gap_rows_worker_process(i0: int, i1: int) -> Tuple[int, np.ndarray]:
    gammas = np.asarray(_G["gammas"], dtype=float)  # type: ignore[arg-type]
    kappas = np.asarray(_G["kappas"], dtype=float)  # type: ignore[arg-type]
    C_L = _G["C_L"]
    L_base = _G["L_base"]
    D_deph = _G["D_deph"]
    gap_method = str(_G["gap_method"])
    gap_k = int(_G["gap_k"])
    gap_tol = float(_G["gap_tol"])
    is_sparse = bool(_G["is_sparse"])

    block = np.zeros((i1 - i0, len(gammas)), dtype=float)

    for bi, i in enumerate(range(i0, i1)):
        kappa = float(kappas[i])
        if is_sparse:
            L_k = L_base + kappa * D_deph  # type: ignore[operator]
            for j, gamma in enumerate(gammas):
                L = (L_k + float(gamma) * C_L)  # type: ignore[operator]
                kk = int(max(2, min(gap_k, L.shape[0] - 2))) if L.shape[0] > 4 else 2
                try:
                    evals = eigs(L, k=kk, which="LR", return_eigenvectors=False)
                    block[bi, j] = liouvillian_spectral_gap_from_eigs(np.asarray(evals), tol=gap_tol)
                except Exception:
                    evals = la.eigvals(L.toarray())
                    block[bi, j] = liouvillian_spectral_gap_from_eigs(evals, tol=gap_tol)
        else:
            L_k = L_base + kappa * D_deph  # type: ignore[operator]
            for j, gamma in enumerate(gammas):
                L = (L_k + float(gamma) * C_L)  # type: ignore[operator]
                if gap_method == "sparse":
                    # Even if base is dense, allow ARPACK on csr for speed.
                    Ls = csr_matrix(L)
                    kk = int(max(2, min(gap_k, Ls.shape[0] - 2))) if Ls.shape[0] > 4 else 2
                    try:
                        evals = eigs(Ls, k=kk, which="LR", return_eigenvectors=False)
                        block[bi, j] = liouvillian_spectral_gap_from_eigs(np.asarray(evals), tol=gap_tol)
                        continue
                    except Exception:
                        pass
                evals = la.eigvals(np.asarray(L))
                block[bi, j] = liouvillian_spectral_gap_from_eigs(evals, tol=gap_tol)

    return i0, block


def _gap_rows_worker_thread(
    i0: int,
    i1: int,
    gammas: np.ndarray,
    kappas: np.ndarray,
    C_L: object,
    L_base: object,
    D_deph: object,
    *,
    gap_method: str,
    gap_k: int,
    gap_tol: float,
    is_sparse: bool,
) -> Tuple[int, np.ndarray]:
    block = np.zeros((i1 - i0, len(gammas)), dtype=float)

    for bi, i in enumerate(range(i0, i1)):
        kappa = float(kappas[i])
        if is_sparse:
            L_k = L_base + kappa * D_deph  # type: ignore[operator]
            for j, gamma in enumerate(gammas):
                L = (L_k + float(gamma) * C_L)  # type: ignore[operator]
                kk = int(max(2, min(gap_k, L.shape[0] - 2))) if L.shape[0] > 4 else 2
                try:
                    evals = eigs(L, k=kk, which="LR", return_eigenvectors=False)
                    block[bi, j] = liouvillian_spectral_gap_from_eigs(np.asarray(evals), tol=gap_tol)
                except Exception:
                    evals = la.eigvals(L.toarray())
                    block[bi, j] = liouvillian_spectral_gap_from_eigs(evals, tol=gap_tol)
        else:
            L_k = L_base + kappa * D_deph  # type: ignore[operator]
            for j, gamma in enumerate(gammas):
                L = (L_k + float(gamma) * C_L)  # type: ignore[operator]
                if gap_method == "sparse":
                    Ls = csr_matrix(L)
                    kk = int(max(2, min(gap_k, Ls.shape[0] - 2))) if Ls.shape[0] > 4 else 2
                    try:
                        evals = eigs(Ls, k=kk, which="LR", return_eigenvectors=False)
                        block[bi, j] = liouvillian_spectral_gap_from_eigs(np.asarray(evals), tol=gap_tol)
                        continue
                    except Exception:
                        pass
                evals = la.eigvals(np.asarray(L))
                block[bi, j] = liouvillian_spectral_gap_from_eigs(evals, tol=gap_tol)

    return i0, block


def compute_gap_heatmap(
    gammas: np.ndarray,
    kappas: np.ndarray,
    C_L: object,
    L_base: object,
    D_deph: object,
    *,
    is_sparse: bool,
    gap_method: Literal["dense", "sparse"] = "dense",
    gap_k: int = 12,
    gap_tol: float = 1e-10,
    jobs: int = 1,
    executor: Literal["threads", "processes"] = "threads",
    chunk_rows: int = 0,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute Liouvillian gap heatmap over (kappa, gamma)."""
    gammas = np.asarray(gammas, dtype=float)
    kappas = np.asarray(kappas, dtype=float)

    nk = int(len(kappas))
    ng = int(len(gammas))
    out = np.zeros((nk, ng), dtype=float)

    jobs = int(max(1, jobs))
    if chunk_rows <= 0:
        target_tasks = max(jobs * 4, 1)
        chunk_rows = max(1, int(math.ceil(nk / target_tasks)))

    tasks: list[tuple[int, int]] = [(i0, min(i0 + chunk_rows, nk)) for i0 in range(0, nk, chunk_rows)]
    total_tasks = len(tasks)

    prog = None
    if show_progress:
        prog = tqdm(total=total_tasks, desc="Fig4b gap heatmap", unit="chunk") if tqdm else _SimpleProgress(total_tasks, "Fig4b gap heatmap")

    if jobs == 1:
        for i0, i1 in tasks:
            _, block = _gap_rows_worker_thread(
                i0,
                i1,
                gammas,
                kappas,
                C_L,
                L_base,
                D_deph,
                gap_method=str(gap_method),
                gap_k=int(gap_k),
                gap_tol=float(gap_tol),
                is_sparse=bool(is_sparse),
            )
            out[i0:i1, :] = block
            if prog:
                prog.update(1)
        if prog and hasattr(prog, "close"):
            prog.close()
        return out

    if executor == "processes":
        try:
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_init_gap_worker,
                initargs=(gammas, kappas, C_L, L_base, D_deph, str(gap_method), int(gap_k), float(gap_tol), bool(is_sparse)),
            ) as ex:
                futures = {ex.submit(_gap_rows_worker_process, i0, i1): (i0, i1) for (i0, i1) in tasks}
                for fut in as_completed(futures):
                    i0, block = fut.result()
                    out[i0:i0 + block.shape[0], :] = block
                    if prog:
                        prog.update(1)
        except Exception as e:  # pragma: no cover
            print(f"Warning: process executor failed ({type(e).__name__}: {e}); falling back to threads.")
            executor = "threads"

    if executor == "threads":
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {
                ex.submit(
                    _gap_rows_worker_thread,
                    i0,
                    i1,
                    gammas,
                    kappas,
                    C_L,
                    L_base,
                    D_deph,
                    gap_method=str(gap_method),
                    gap_k=int(gap_k),
                    gap_tol=float(gap_tol),
                    is_sparse=bool(is_sparse),
                ): (i0, i1)
                for (i0, i1) in tasks
            }
            for fut in as_completed(futures):
                i0, block = fut.result()
                out[i0:i0 + block.shape[0], :] = block
                if prog:
                    prog.update(1)

    if prog and hasattr(prog, "close"):
        prog.close()

    return out


# =========
# Plotting
# =========

def _save_fig(fig: plt.Figure, out_dir: Path, base: str, dpi: int = 300) -> None:
    fig.savefig(out_dir / f"{base}_{dpi}dpi.png", dpi=dpi)
    fig.savefig(out_dir / f"{base}.svg")


def _heatmap_extent(x: np.ndarray, y: np.ndarray) -> list[float]:
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    return [float(x[0] - 0.5 * dx), float(x[-1] + 0.5 * dx), float(y[0] - 0.5 * dy), float(y[-1] + 0.5 * dy)]


def plot_landscape(
    grid: np.ndarray,
    gammas: np.ndarray,
    kappas: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    annotate_max: bool = True,
) -> plt.Figure:
    extent = _heatmap_extent(gammas, kappas)

    i_max, j_max = np.unravel_index(np.nanargmax(grid), grid.shape)
    max_kappa = float(kappas[i_max])
    max_gamma = float(gammas[j_max])
    max_val = float(grid[i_max, j_max])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", extent=extent, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if annotate_max:
        ax.scatter([max_gamma], [max_kappa], s=80, marker="o", edgecolors="k", linewidths=1.0)
        gx_mid = 0.5 * (extent[0] + extent[1])
        ky_mid = 0.5 * (extent[2] + extent[3])

        dx = -10 if max_gamma >= gx_mid else 10
        ha = "right" if dx < 0 else "left"
        dy = -10 if max_kappa >= ky_mid else 10
        va = "top" if dy < 0 else "bottom"

        label = f"max={max_val:.3g}\n(γ={max_gamma:.2f}, κ={max_kappa:.2f})"
        ax.annotate(
            label,
            xy=(max_gamma, max_kappa),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.3", alpha=0.9),
            arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.8),
        )

    pad_g = max(0.08 * (gammas[-1] - gammas[0]), 1e-9)
    pad_k = max(0.08 * (kappas[-1] - kappas[0]), 1e-9)
    ax.set_xlim(extent[0] - pad_g, extent[1] + pad_g)
    ax.set_ylim(extent[2] - pad_k, extent[3] + pad_k)

    plt.tight_layout()
    return fig


# =========
# CLI / main
# =========

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate QRN Figures 2–4 (+ optional Fig4b gap heatmap).")
    p.add_argument("--out-dir", default=".", help="Output directory (default: current directory).")
    p.add_argument("--N", type=int, default=DEFAULT_N, help=f"Chain length without sink (default: {DEFAULT_N}).")
    p.add_argument("--T", type=float, default=DEFAULT_T_HORIZON, help=f"Time horizon (default: {DEFAULT_T_HORIZON}).")
    p.add_argument("--eta", type=float, default=DEFAULT_ETA, help=f"Sink rate eta (default: {DEFAULT_ETA}).")
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help=f"Gamma used for Fig2/Fig3 (default: {DEFAULT_GAMMA}).")
    p.add_argument("--kappa-q", type=float, default=1.0, help="QRN kappa for Fig2/Fig3 (default: 1.0).")

    # Fig4 grid
    p.add_argument("--gamma-min", type=float, default=0.5, help="Fig4: min gamma (default: 0.5).")
    p.add_argument("--gamma-max", type=float, default=4.0, help="Fig4: max gamma (default: 4.0).")
    p.add_argument("--kappa-min", type=float, default=0.0, help="Fig4: min kappa (default: 0.0).")
    p.add_argument("--kappa-max", type=float, default=2.5, help="Fig4: max kappa (default: 2.5).")
    p.add_argument("--heatmap-ng", type=int, default=20, help="Fig4: number of gamma grid points (default: 20).")
    p.add_argument("--heatmap-nk", type=int, default=20, help="Fig4: number of kappa grid points (default: 20).")

    # Parallel
    p.add_argument("--jobs", type=int, default=1, help="Parallel workers for heatmaps (default: 1).")
    p.add_argument("--executor", choices=["threads", "processes"], default="threads", help="Parallel backend (default: threads).")
    p.add_argument("--chunk-rows", type=int, default=0, help="Rows per task chunk; 0=auto (default: 0).")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar/ETA.")

    # Gap heatmap
    p.add_argument("--gap-heatmap", action="store_true", help="Also compute Fig4b Liouvillian gap heatmap.")
    p.add_argument("--gap-method", choices=["dense", "sparse"], default="sparse", help="Gap eigen-solver (default: sparse).")
    p.add_argument("--gap-k", type=int, default=12, help="ARPACK: number of eigenvalues requested (default: 12).")
    p.add_argument("--gap-tol", type=float, default=1e-10, help="Tolerance for ignoring steady modes (default: 1e-10).")

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    n = int(args.N)
    d = n + 1
    sink = n
    target = n - 1

    # Reproducible potential (matches earlier script style, but depends on N)
    rng = np.random.default_rng(42)
    epsilon = rng.uniform(-0.5, 0.5, n)
    V_pot = 0.3 * (np.arange(n)[::-1] + epsilon)
    V_pot = V_pot - np.min(V_pot)
    V_diag = np.diag(V_pot)

    # Initial state: |0><0|
    rho0 = np.zeros((d, d), dtype=complex)
    rho0[0, 0] = 1.0 + 0j

    # Precompute dense components for dynamics + efficiency heatmap
    C_L, C_V, D_deph, D_sink = precompute_liouvillian_components(n, V_pot, target, sparse=False)
    L_base = C_V + float(args.eta) * D_sink

    def L_super(gamma: float, kappa: float) -> np.ndarray:
        return L_base + float(gamma) * C_L + float(kappa) * D_deph

    times = np.linspace(0.0, float(args.T), 401)

    # --------
    # Figure 2
    # --------
    rhos_coh = propagate_density_timeseries(L_super(float(args.gamma), 0.0), rho0, times)
    rhos_qrn = propagate_density_timeseries(L_super(float(args.gamma), float(args.kappa_q)), rho0, times)

    Q = crw_generator_chain_with_sink(n, float(args.gamma), float(args.eta), target)
    p0 = np.zeros((d,), dtype=float)
    p0[0] = 1.0
    ps_crw = propagate_prob_timeseries(Q, p0, times)

    p_sink_coh = np.real(rhos_coh[:, sink, sink])
    p_sink_qrn = np.real(rhos_qrn[:, sink, sink])
    p_sink_crw = ps_crw[:, sink]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(times, p_sink_coh, label="Coherent ($\\kappa=0$)", linestyle="--", linewidth=2)
    plt.plot(times, p_sink_qrn, label=f"QRN Netting ($\\kappa={float(args.kappa_q):g}$)", linewidth=3)
    plt.plot(times, p_sink_crw, label="Classical RW", linestyle=":", linewidth=2, alpha=0.8)
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Success probability $P_{success}(t)$")
    plt.title("Figure 2: Targeting / Absorption")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    _save_fig(fig, out_dir, "Fig2_QRN_Targeting")
    plt.close(fig)

    # --------
    # Figure 3
    # --------
    ev_coh = np.array([expected_potential_conditional(r, V_diag) for r in rhos_coh], dtype=float)
    ev_qrn = np.array([expected_potential_conditional(r, V_diag) for r in rhos_qrn], dtype=float)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(times, ev_coh, label="Coherent ($\\kappa=0$)", linestyle="--", linewidth=2)
    plt.plot(times, ev_qrn, label=f"QRN Netting ($\\kappa={float(args.kappa_q):g}$)", linewidth=3)
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Expected Potential (conditional on not-yet-readout)")
    plt.title("Figure 3: Expected Potential")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    _save_fig(fig, out_dir, "Fig3_QRN_ExpectedPotential")
    plt.close(fig)

    # -----------------
    # Figure 4: heatmap
    # -----------------
    gammas = np.linspace(float(args.gamma_min), float(args.gamma_max), int(args.heatmap_ng))
    kappas = np.linspace(float(args.kappa_min), float(args.kappa_max), int(args.heatmap_nk))

    v0 = vecF(rho0)
    sink_vec_index = sink + sink * d  # index of (sink, sink) in vecF

    heatmap = compute_efficiency_heatmap(
        gammas,
        kappas,
        C_L,
        L_base,
        D_deph,
        float(args.T),
        v0,
        int(sink_vec_index),
        jobs=int(args.jobs),
        executor=str(args.executor),
        chunk_rows=int(args.chunk_rows),
        show_progress=not bool(args.no_progress),
    )

    fig = plot_landscape(
        heatmap,
        gammas,
        kappas,
        title="Figure 4: Efficiency Landscape",
        xlabel="Diffusion / Connectivity $\\gamma$",
        ylabel="Dephasing / Noise $\\kappa$",
        cbar_label="$P_{success}(T)$",
    )
    _save_fig(fig, out_dir, "Fig4_QRN_EfficiencyLandscape")
    plt.close(fig)

    # --------------------------
    # Figure 4b: gap heatmap
    # --------------------------
    if bool(args.gap_heatmap):
        # Use sparse components for ARPACK when possible (bigger N).
        use_sparse_base = (str(args.gap_method) == "sparse")
        C_Lg, C_Vg, D_deph_g, D_sink_g = precompute_liouvillian_components(n, V_pot, target, sparse=use_sparse_base)
        L_base_g = C_Vg + float(args.eta) * D_sink_g

        gap_map = compute_gap_heatmap(
            gammas,
            kappas,
            C_Lg,
            L_base_g,
            D_deph_g,
            is_sparse=bool(use_sparse_base),
            gap_method=str(args.gap_method),
            gap_k=int(args.gap_k),
            gap_tol=float(args.gap_tol),
            jobs=int(args.jobs),
            executor=str(args.executor),
            chunk_rows=int(args.chunk_rows),
            show_progress=not bool(args.no_progress),
        )

        fig = plot_landscape(
            gap_map,
            gammas,
            kappas,
            title="Figure 4b: Liouvillian Spectral Gap",
            xlabel="Diffusion / Connectivity $\\gamma$",
            ylabel="Dephasing / Noise $\\kappa$",
            cbar_label="Gap $g$",
        )
        _save_fig(fig, out_dir, "Fig4b_QRN_LiouvillianGapLandscape")
        plt.close(fig)

    print(f"Done. Output in: {out_dir}")


if __name__ == "__main__":
    try:
        import multiprocessing as mp

        mp.freeze_support()
    except Exception:
        pass
    main()
