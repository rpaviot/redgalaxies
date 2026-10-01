"""Truncation-aware, pairwise fit of the intrinsic colour-colour correlation r_ij(z).

redMaPPer (Rykoff+14, Sect. "Measuring C_jk^int(z)") fits the off-diagonal
intrinsic covariance C_int_ij = r_ij(z) c_i(z) c_j(z) two colours at a time,
with a/b/c(z) frozen, in a priority order, rejecting any step that makes the
TOTAL intrinsic covariance non positive-definite.  This module does the same,
but on our two-stage step-1 selection, which truncates the sample in all four
colours at once.  Per galaxy and pair P = (i, j) the likelihood is

    L = N2(x_P; model_P, S) * B(x_P) / P_surv,    S = C_int,P(r_ij) + C_err,P

  * N2 -- the 2-D Gaussian of the red-sequence model;
  * P_surv = int S_sel(u) N2(u; d, S) d^2u -- the mass of that Gaussian that
    survives the step-1 cut (the 2-D analogue of the trunc2 term), with
    u = x_P - mu2_P and d = model_P - mu2_P;
  * S_sel(u) = box(u) * F_chi2_2(chi2_4 - u^T G_PP^-1 u), G = V2 + C_err:
    the stage-2 chi^2_4 ellipsoid marginalised over the other two colours
    (their conditional Mahalanobis is chi^2_2), times the stage-1 box on the
    reference colour.  If the reference colour is IN the pair the box is an
    integration limit; if it is not, the box still cuts the pair through the
    correlations r(ref, i), r(ref, j) (the "leakage" that over-corrected the
    i-z slope in the rho=0.6 mock) and enters as
        B(x_P) = P(|x_ref - mu1| < t1 | x_P)
    -- an erf in the conditional Gaussian of the reference colour, which
    depends on r_ij too, so it appears in the numerator AND inside P_surv.
    Those two r's come from earlier pairs / an earlier pass.

The survival integral is a 2-D Gauss-Legendre quadrature on the selection
region itself: axis 0 (the reference colour when it is in the pair) is mapped
u0 = mid + half sin(phi) over the box/ellipse interval, and axis 1 over the
ellipse chord at that u0, u1 = cm + h sin(psi); the sine maps absorb the
square-root edges of the ellipse, and on this grid the chi^2_2 taper is
1 - exp(-(chi2_4 - q0) cos^2(psi) / 2).  All of that depends only on the
FIXED window, so abscissae and weights are built once per pair.

Unit check (``check_normalisation``): with S = G, d = 0 and no box the survival
mass must equal F_chi2_4(chi2_4).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.special import ndtr

COLOURS = ["gr", "ri", "iz", "ug"]
# Best-constrained first (redMaPPer); pairs that contain a reference colour
# (gr below the switch, ri above) come first so later pairs' box leakage
# uses already-fitted r(ref, .).
DEFAULT_PAIR_ORDER = [("gr", "ri"), ("ri", "iz"), ("gr", "iz"),
                      ("gr", "ug"), ("ri", "ug"), ("iz", "ug")]
LOG_2PI = np.log(2.0 * np.pi)


def pair_key(a: str, b: str) -> Tuple[str, str]:
    """Canonical (colour order) key for a pair."""
    return (a, b) if COLOURS.index(a) < COLOURS.index(b) else (b, a)


# --------------------------------------------------------------- window ----
def load_window_full(path: str) -> Dict[str, np.ndarray]:
    """Two-stage window npz WITH the full stage-2 matrices (V2full, mu2full).

    Written by src/build_twostage_truncation.py (real step 1) or
    src/validate_twostage_mock.py (mocks).  Colour order is ``colors``.
    """
    d = np.load(path, allow_pickle=True)
    if "ndim" in d.files and np.any(np.asarray(d["ndim"], float) != 4):
        raise NotImplementedError(f"{path} has 3-colour stage-2 bins (u-g dropped); the pairwise "
                                  "r fit assumes 4 colours (F_chi2_2 taper) -- not generalised yet")
    if "V2full" not in d.files:
        raise ValueError(f"{path} has no V2full/mu2full: rebuild it with "
                         "src/build_twostage_truncation.py (it now stores the "
                         "full stage-2 matrices needed for the pairwise fit)")
    cols = [str(c) for c in d["colors"]]
    if cols != COLOURS:
        raise ValueError(f"window colour order {cols} != {COLOURS}")
    is_ref = np.stack([np.asarray(d[f"{c}_is_ref"], float) for c in cols], axis=1)
    ref = np.argmax(is_ref, axis=1)
    rows = np.arange(len(ref))
    get = lambda k: np.stack([np.asarray(d[f"{c}_{k}"], float) for c in cols], 1)[rows, ref]  # noqa: E731
    return {"z_edges": np.asarray(d["z_edges"], float),
            "V2full": np.asarray(d["V2full"], float),
            "mu2full": np.asarray(d["mu2full"], float),
            "chi2_4": np.asarray(d[f"{cols[0]}_chi2_4"], float),
            "ref": ref, "mu1": get("mu1"), "V1": get("V1"), "chi2_1": get("chi2_1")}


class XcovData:
    """Per-galaxy inputs, all in COLOURS order.

    X, model, c : (N, 4) observed colours, frozen a + b (m - m_ref), c(z)
    E           : (N, 4, 4) photometric colour covariance (shared bands)
    z           : (N,)
    """

    def __init__(self, X, E, z, model, c, window: Dict[str, np.ndarray]):
        self.X, self.E, self.z = np.asarray(X, float), np.asarray(E, float), np.asarray(z, float)
        self.model, self.c = np.asarray(model, float), np.abs(np.asarray(c, float))
        self.N = len(self.z)
        edges = window["z_edges"]
        idx = np.clip(np.searchsorted(edges, self.z, side="right") - 1, 0, len(edges) - 2)
        n = np.arange(self.N)
        self.mu2 = window["mu2full"][idx]
        self.G = window["V2full"][idx] + self.E
        self.k4 = window["chi2_4"][idx]
        self.ref = window["ref"][idx]
        self.mu1 = window["mu1"][idx]
        self.t1 = np.sqrt(window["chi2_1"][idx] * (window["V1"][idx] + self.E[n, self.ref, self.ref]))


# ------------------------------------------------------------ r splines ----
class RSet:
    """Current r(z) node values for every pair (0 where not yet fitted)."""

    def __init__(self, nodes: np.ndarray, r_bound: float = 0.95):
        self.nodes = np.asarray(nodes, float)
        self.r_bound = r_bound
        self.vals: Dict[Tuple[str, str], np.ndarray] = {}

    def get(self, a: str, b: str) -> np.ndarray:
        return self.vals.get(pair_key(a, b), np.zeros(len(self.nodes)))

    def set(self, a: str, b: str, v: np.ndarray) -> None:
        self.vals[pair_key(a, b)] = np.asarray(v, float)

    def eval(self, a: str, b: str, z: np.ndarray, vals: Optional[np.ndarray] = None) -> np.ndarray:
        v = self.get(a, b) if vals is None else vals
        # Same CubicSpline the photo-z step builds (make_r_cross_splines), but
        # clamped in z and in value so the likelihood never sees |r| >= 1.
        z = np.clip(z, self.nodes[0], self.nodes[-1])
        return np.clip(CubicSpline(self.nodes, v)(z), -0.99, 0.99)


def min_eig_cint(rset: RSet, c_funcs: Dict[str, Callable], zgrid: np.ndarray,
                 override: Optional[Tuple[str, str, np.ndarray]] = None) -> np.ndarray:
    """Smallest eigenvalue of the full 4x4 intrinsic covariance on ``zgrid``."""
    cz = np.stack([np.abs(c_funcs[c](zgrid)) for c in COLOURS], 1)          # (Z, 4)
    R = np.tile(np.eye(4), (len(zgrid), 1, 1))
    for a in range(4):
        for b in range(a + 1, 4):
            ca, cb = COLOURS[a], COLOURS[b]
            v = override[2] if override is not None and pair_key(ca, cb) == override[:2] else None
            R[:, a, b] = R[:, b, a] = rset.eval(ca, cb, zgrid, v)
    C = R * cz[:, :, None] * cz[:, None, :]
    return np.linalg.eigvalsh(C)[:, 0]


# ------------------------------------------------------- quadrature ------
def _taper(x: np.ndarray, df_rest: int) -> np.ndarray:
    """F_chi2_{df_rest}(x) for x >= 0 (0 below)."""
    x = np.clip(x, 0.0, None)
    if df_rest == 2:
        return 1.0 - np.exp(-0.5 * x)
    from scipy.stats import chi2
    return chi2.cdf(x, df=df_rest)


def ellipse_box_quadrature(G00, G11, G01, k, lo, hi, n_quad: int = 12, df_rest: int = 2):
    """Nodes and weights for int_{ellipse, lo<u0<hi} F_{df_rest}(k - u^T G^-1 u) f(u) d^2u.

    u0 is mapped mid + half sin(phi) over [lo, hi] (already clipped to the
    ellipse extent), u1 over the chord at u0 as cm + h sin(psi); the sine maps
    absorb the square-root edges.  Returns U0, U1, W of shape (N, n_quad^2),
    W including the Jacobians and the taper, so the integral is sum(W f(U0, U1)).
    """
    x, w = np.polynomial.legendre.leggauss(n_quad)
    ang, wang = 0.5 * np.pi * x, 0.5 * np.pi * w
    hi = np.maximum(hi, lo)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    u0 = mid[:, None] + half[:, None] * np.sin(ang)[None, :]
    jac0 = half[:, None] * np.cos(ang)[None, :] * wang[None, :]
    rem = np.clip(k[:, None] - u0 ** 2 / G00[:, None], 0.0, None)
    s2 = G11 - G01 ** 2 / G00
    h = np.sqrt(s2[:, None] * rem)
    cm = (G01 / G00)[:, None] * u0
    u1 = cm[:, :, None] + h[:, :, None] * np.sin(ang)[None, None, :]
    jac1 = h[:, :, None] * np.cos(ang)[None, None, :] * wang[None, None, :]
    taper = _taper(rem[:, :, None] * np.cos(ang)[None, None, :] ** 2, df_rest)
    N = len(k)
    W = (jac0[:, :, None] * jac1 * taper).reshape(N, -1)
    U0 = np.broadcast_to(u0[:, :, None], u1.shape).reshape(N, -1)
    return U0, u1.reshape(N, -1), W


def chord_general(cm, s2, rem, lo, hi, n_quad: int = 16, df_rest: int = 2):
    """Nodes/weights for int F_{df_rest}(rem - (u - cm)^2 / s2) f(u) du over the
    chord |u - cm| < sqrt(s2 rem) cut to [lo, hi]. Returns U, W of shape (N, n_quad)."""
    x, w = np.polynomial.legendre.leggauss(n_quad)
    ang, wang = 0.5 * np.pi * x, 0.5 * np.pi * w
    rem = np.clip(rem, 0.0, None)
    h = np.sqrt(s2 * rem)
    a = np.maximum(cm - h, lo)
    b = np.maximum(np.minimum(cm + h, hi), a)
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    U = mid[:, None] + half[:, None] * np.sin(ang)[None, :]
    jac = half[:, None] * np.cos(ang)[None, :] * wang[None, :]
    q = rem[:, None] - (U - cm[:, None]) ** 2 / s2[:, None]
    return U, jac * _taper(q, df_rest)


def ellipsoid3_box_quadrature(G, k, lo, hi, n_quad: int = 8):
    """3-D analogue of ellipse_box_quadrature for axes (0, 1, 2) of G (N, 3, 3):
    int_{ellipsoid, lo<u0<hi} F_chi2_1(k - u^T G^-1 u) f(u) d^3u.
    u0 on [lo, hi]; (u1, u2) on the conditional ellipse at u0, nested chords.
    Returns U (N, K, 3) and W (N, K), K = n_quad^3."""
    x, w = np.polynomial.legendre.leggauss(n_quad)
    ang, wang = 0.5 * np.pi * x, 0.5 * np.pi * w
    sa, ca = np.sin(ang), np.cos(ang)
    N = len(k)
    hi = np.maximum(hi, lo)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    u0 = mid[:, None] + half[:, None] * sa[None, :]                          # (N, n)
    j0 = half[:, None] * ca[None, :] * wang[None, :]
    G00 = G[:, 0, 0]
    rem0 = np.clip(k[:, None] - u0 ** 2 / G00[:, None], 0.0, None)
    # (u1, u2) | u0 ~ G-conditional
    m1 = (G[:, 1, 0] / G00)[:, None] * u0
    m2 = (G[:, 2, 0] / G00)[:, None] * u0
    C11 = G[:, 1, 1] - G[:, 1, 0] ** 2 / G00
    C22 = G[:, 2, 2] - G[:, 2, 0] ** 2 / G00
    C12 = G[:, 1, 2] - G[:, 1, 0] * G[:, 2, 0] / G00
    h1 = np.sqrt(C11[:, None] * rem0)
    u1 = m1[:, :, None] + h1[:, :, None] * sa[None, None, :]                # (N, n, n)
    j1 = h1[:, :, None] * ca[None, None, :] * wang[None, None, :]
    rem1 = rem0[:, :, None] * ca[None, None, :] ** 2
    m2c = m2[:, :, None] + (C12 / C11)[:, None, None] * (u1 - m1[:, :, None])
    C2c = C22 - C12 ** 2 / C11
    h2 = np.sqrt(C2c[:, None, None] * rem1)
    u2 = m2c[..., None] + h2[..., None] * sa                                # (N, n, n, n)
    j2 = h2[..., None] * ca * wang
    taper = _taper(rem1[..., None] * ca ** 2, 1)
    W = (j0[:, :, None, None] * j1[..., None] * j2 * taper).reshape(N, -1)
    U = np.stack([np.broadcast_to(u0[:, :, None, None], u2.shape),
                  np.broadcast_to(u1[..., None], u2.shape), u2], -1).reshape(N, -1, 3)
    return U, W


def chord_quadrature(u_fix, Gff, Goo, Gfo, k, lo, hi, n_quad: int = 16, df_rest: int = 2):
    """Nodes/weights on the OTHER axis at a fixed u_fix, for
    int F_{df_rest}(k - q(u_fix, u)) f(u) du over the ellipse chord cut to [lo, hi].
    Returns U (N, n_quad), W (N, n_quad) with Jacobian and taper."""
    x, w = np.polynomial.legendre.leggauss(n_quad)
    ang, wang = 0.5 * np.pi * x, 0.5 * np.pi * w
    s2 = Goo - Gfo ** 2 / Gff
    cm = Gfo / Gff * u_fix
    rem = np.clip(k - u_fix ** 2 / Gff, 0.0, None)
    h = np.sqrt(s2 * rem)
    a = np.maximum(cm - h, lo)
    b = np.maximum(np.minimum(cm + h, hi), a)
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    U = mid[:, None] + half[:, None] * np.sin(ang)[None, :]
    jac = half[:, None] * np.cos(ang)[None, :] * wang[None, :]
    q = rem[:, None] - (U - cm[:, None]) ** 2 / s2[:, None]
    return U, jac * _taper(q, df_rest)


# ------------------------------------------------------------ one pair -----
class PairLikelihood:
    """-2 log L for r_ij(z) node values; see the module docstring."""

    def __init__(self, data: XcovData, ci: str, cj: str, rset: RSet,
                 c_funcs: Dict[str, Callable], n_quad: int = 12, n_quad3: int = 12,
                 truncation: bool = True, prior_width: float = 0.45,
                 eig_floor: float = 1e-4, exact: bool = False):
        self.data, self.ci, self.cj = data, ci, cj
        self.exact = exact
        self.rset, self.c_funcs = rset, c_funcs
        self.truncation, self.prior_width, self.eig_floor = truncation, prior_width, eig_floor
        i, j = COLOURS.index(ci), COLOURS.index(cj)
        self.i, self.j = i, j
        D = data
        n = np.arange(D.N)
        self.delta = (D.X - D.model)[:, [i, j]]                      # (N, 2)
        self.Ei, self.Ej, self.Eij = D.E[:, i, i], D.E[:, j, j], D.E[:, i, j]
        self.c_i, self.c_j = D.c[:, i], D.c[:, j]
        self.zgrid = np.linspace(rset.nodes[0], rset.nodes[-1], 46)
        if not truncation:
            return

        has_ref = (D.ref == i) | (D.ref == j)
        swap = D.ref == j                   # put the reference colour on axis 0
        a0 = np.where(swap, j, i)
        a1 = np.where(swap, i, j)
        G00, G11, G01 = D.G[n, a0, a0], D.G[n, a1, a1], D.G[n, a0, a1]
        k4 = D.k4
        ext = np.sqrt(k4 * G00)
        lo, hi = -ext.copy(), ext.copy()
        off = D.mu1 - D.mu2[n, D.ref]       # box centre relative to the ellipse centre
        lo = np.where(has_ref, np.maximum(lo, off - D.t1), lo)
        hi = np.where(has_ref, np.minimum(hi, off + D.t1), hi)
        hi = np.maximum(hi, lo)             # empty intersection -> zero mass

        U0, U1, W = ellipse_box_quadrature(G00, G11, G01, k4, lo, hi, n_quad)
        # back to (i, j) order, then offset by the model: v = u - d
        Ui = np.where(swap[:, None], U1, U0)
        Uj = np.where(swap[:, None], U0, U1)
        d = (D.model - D.mu2)[:, [i, j]]
        self.Vi = Ui - d[:, [0]]
        self.Vj = Uj - d[:, [1]]
        self.W = W

        # Box leakage for galaxies whose reference colour is NOT in the pair.
        self.leak = np.where(~has_ref)[0]
        if len(self.leak):
            L = self.leak
            r = D.ref[L]
            self.L_Err = D.E[L, r, r]
            self.L_Eri = D.E[L, r, i]
            self.L_Erj = D.E[L, r, j]
            self.L_cr = D.c[L, r]
            self.L_mref = D.model[L, r]
            self.L_lo = D.mu1[L] - D.t1[L]
            self.L_hi = D.mu1[L] + D.t1[L]
            # r(ref, i), r(ref, j) are fixed during this pair's fit
            zL = D.z[L]
            self.L_rri = np.zeros(len(L))
            self.L_rrj = np.zeros(len(L))
            for k, cr in enumerate(COLOURS):
                mk = r == k
                if not mk.any():
                    continue
                if k != i:
                    self.L_rri[mk] = rset.eval(cr, ci, zL[mk])
                if k != j:
                    self.L_rrj[mk] = rset.eval(cr, cj, zL[mk])
            if exact:
                # EXACT: integrate the reference colour out in 3-D (ref, i, j),
                # the other colour's conditional Mahalanobis is chi^2_1. The
                # product B x F_chi2_2 double counts (the F taper assumes an
                # untruncated reference colour).
                nL = len(L)
                ax = np.stack([r, np.full(nL, i), np.full(nL, j)], 1)        # (nL, 3)
                G3 = D.G[L[:, None, None], ax[:, :, None], ax[:, None, :]]
                k4L = D.k4[L]
                off = D.mu1[L] - D.mu2[L, r]
                ext = np.sqrt(k4L * G3[:, 0, 0])
                U3, self.W3 = ellipsoid3_box_quadrature(
                    G3, k4L, np.maximum(-ext, off - D.t1[L]), np.minimum(ext, off + D.t1[L]),
                    n_quad=n_quad3)
                d3 = D.model[L[:, None], ax] - D.mu2[L[:, None], ax]
                # 12^3 nodes per galaxy: store in float32 (the integrand is
                # smooth; float32 nodes change P by <1e-6), evaluate in chunks
                self.V3 = (U3 - d3[:, None, :]).astype(np.float32)
                self.W3 = self.W3.astype(np.float32)
                del U3
                # numerator: chord in u_ref at the OBSERVED (u_i, u_j)
                uP = D.X[L][:, [i, j]] - D.mu2[L][:, [i, j]]
                GPP = G3[:, 1:, 1:]
                GrP = G3[:, 0, 1:]
                A = np.linalg.solve(GPP, GrP[..., None])[..., 0]            # G_PP^-1 G_Pr
                cm = np.sum(A * uP, 1)
                s2 = G3[:, 0, 0] - np.sum(A * GrP, 1)
                qP = np.einsum('na,nab,nb->n', uP, np.linalg.inv(GPP), uP)
                self.Ur_num, self.Wr_num = chord_general(
                    cm, s2, k4L - qP, off - D.t1[L], off + D.t1[L], n_quad=16, df_rest=1)
                self.dr3 = d3[:, 0]

    # ------------------------------------------------------------------
    def components(self, theta: np.ndarray):
        """Per-galaxy (log N2, log B_num, log P_surv) at node values theta."""
        D = self.data
        rz = self.rset.eval(self.ci, self.cj, D.z, theta)
        S00 = self.c_i ** 2 + self.Ei
        S11 = self.c_j ** 2 + self.Ej
        S01 = rz * self.c_i * self.c_j + self.Eij
        det = S00 * S11 - S01 ** 2
        if np.any(det <= 0):
            return None
        d0, d1 = self.delta[:, 0], self.delta[:, 1]
        chi2 = (S11 * d0 ** 2 - 2 * S01 * d0 * d1 + S00 * d1 ** 2) / det
        logN = -0.5 * chi2 - 0.5 * np.log(det) - LOG_2PI
        logB = np.zeros(D.N)
        logP = np.zeros(D.N)
        if not self.truncation:
            return logN, logB, logP

        q = (S11[:, None] * self.Vi ** 2 - 2 * S01[:, None] * self.Vi * self.Vj
             + S00[:, None] * self.Vj ** 2) / det[:, None]
        g = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))[:, None]
        integrand = self.W * g
        if len(self.leak):
            L = self.leak
            # conditional of x_ref | x_P under the model: mean m_r + K (x_P - model_P)
            Sri = self.L_rri * self.L_cr * self.c_i[L] + self.L_Eri
            Srj = self.L_rrj * self.L_cr * self.c_j[L] + self.L_Erj
            Srr = self.L_cr ** 2 + self.L_Err
            a, b, cc, dt = S00[L], S01[L], S11[L], det[L]
            K0 = (Sri * cc - Srj * b) / dt
            K1 = (Srj * a - Sri * b) / dt
            s = np.sqrt(np.clip(Srr - K0 * Sri - K1 * Srj, 1e-12, None))

            def box(e0, e1):
                mean = self.L_mref[:, None] + K0[:, None] * e0 + K1[:, None] * e1
                return (ndtr((self.L_hi[:, None] - mean) / s[:, None])
                        - ndtr((self.L_lo[:, None] - mean) / s[:, None]))

            if not self.exact:
                integrand[L] *= box(self.Vi[L], self.Vj[L])
                logB[L] = np.log(np.clip(box(self.delta[L, :1], self.delta[L, 1:])[:, 0],
                                         1e-300, None))
        logP = np.log(np.clip(integrand.sum(axis=1), 1e-300, None))
        if len(self.leak) and self.exact:
            L = self.leak
            S3 = np.empty((len(L), 3, 3))
            S3[:, 0, 0] = Srr
            S3[:, 0, 1] = S3[:, 1, 0] = Sri
            S3[:, 0, 2] = S3[:, 2, 0] = Srj
            S3[:, 1, 1], S3[:, 2, 2] = S00[L], S11[L]
            S3[:, 1, 2] = S3[:, 2, 1] = S01[L]
            sign, ld = np.linalg.slogdet(S3)
            if np.any(sign <= 0):
                return None
            Si = np.linalg.inv(S3).astype(np.float32)
            P3 = np.empty(len(L))
            for a in range(0, len(L), 8192):
                sl = slice(a, a + 8192)
                V = self.V3[sl]
                q = np.einsum('nka,nab,nkb->nk', V, Si[sl], V, optimize=True)
                P3[sl] = np.sum(self.W3[sl] * np.exp(-0.5 * q), 1, dtype=np.float64)
            P3 *= np.exp(-0.5 * ld) / (2 * np.pi) ** 1.5
            logP[L] = np.log(np.clip(P3, 1e-300, None))
            # N(u_ref | x_P) under the model: mean d_r + K delta_P, var s^2
            mean = self.dr3 + K0 * self.delta[L, 0] + K1 * self.delta[L, 1]
            num = np.sum(self.Wr_num * np.exp(-0.5 * (self.Ur_num - mean[:, None]) ** 2
                                              / s[:, None] ** 2), 1) / (np.sqrt(2 * np.pi) * s)
            logB[L] = np.log(np.clip(num, 1e-300, None))
        return logN, logB, logP

    def __call__(self, theta: np.ndarray) -> float:
        comp = self.components(theta)
        if comp is None:
            return 1e30
        logN, logB, logP = comp
        loss = -2.0 * float(np.sum(logN + logB - logP))
        loss += float(np.sum((np.asarray(theta) / self.prior_width) ** 2))
        lam = min_eig_cint(self.rset, self.c_funcs, self.zgrid,
                           override=(*pair_key(self.ci, self.cj), np.asarray(theta)))
        viol = np.clip(self.eig_floor - lam, 0.0, None) / (0.1 * self.eig_floor)
        return loss + 1e3 * float(np.sum(viol ** 2))


def fit_pair(data: XcovData, ci: str, cj: str, rset: RSet, c_funcs, *,
             n_quad: int = 12, truncation: bool = True, prior_width: float = 0.45,
             eig_floor: float = 1e-4, exact: bool = False, n_quad3: int = 12,
             verbose: bool = True) -> Dict:
    """Fit r(ci, cj) node values with every other pair frozen in ``rset``."""
    t0 = time.time()
    like = PairLikelihood(data, ci, cj, rset, c_funcs, n_quad=n_quad,
                          truncation=truncation, prior_width=prior_width,
                          eig_floor=eig_floor, exact=exact, n_quad3=n_quad3)
    x0 = np.clip(rset.get(ci, cj), -0.9, 0.9)
    nll0 = like(x0)
    res = minimize(like, x0, method="L-BFGS-B",
                   bounds=[(-rset.r_bound, rset.r_bound)] * len(x0),
                   options={"maxiter": 500, "eps": 1e-4})
    rset.set(ci, cj, res.x)
    lam = min_eig_cint(rset, c_funcs, like.zgrid)
    out = {"pair": f"{ci}__{cj}", "values": res.x.tolist(), "nll": float(res.fun),
           "nll_start": float(nll0), "success": bool(res.success), "nit": int(res.nit),
           "nfev": int(res.nfev), "min_eig_cint": float(lam.min()),
           "railed": [bool(abs(v) > 0.98 * rset.r_bound) for v in res.x],
           "n_leak": int(len(getattr(like, "leak", []))), "seconds": time.time() - t0}
    if verbose:
        print(f"  r({ci},{cj}) = {np.round(res.x, 3)}  NLL {nll0:.1f} -> {res.fun:.1f} "
              f"nit={res.nit} nfev={res.nfev} min eig={lam.min():.2e} "
              f"[{out['seconds']:.0f}s]", flush=True)
    return out


def fit_all_pairs(data: XcovData, rset: RSet, c_funcs, *,
                  pair_order: Sequence[Tuple[str, str]] = DEFAULT_PAIR_ORDER,
                  n_pass: int = 2, checkpoint: Optional[Callable] = None,
                  **kw) -> List[Dict]:
    """Fit every pair in priority order; repeat ``n_pass`` times so the box
    leakage of early pairs sees the r of later ones. ``checkpoint(log)`` is
    called after every pair."""
    log = []
    for p in range(1, n_pass + 1):
        print(f"--- pass {p}/{n_pass} ---", flush=True)
        for ci, cj in pair_order:
            rec = fit_pair(data, ci, cj, rset, c_funcs, **kw)
            rec["pass"] = p
            log.append(rec)
            if checkpoint is not None:
                checkpoint(log)
    return log


def check_normalisation(data: XcovData, n_quad: int = 12, n_max: int = 2000) -> float:
    """Max |P_surv - F_chi2_4(chi2_4)| with S = G, d = 0, no box (must be ~0).

    Uses the real per-galaxy windows G = V2 + C_err; the model is put at the
    window centre and the stage-1 box is removed, so the stage-2 marginal
    taper integrated against the stage-2 Gaussian itself must give back the
    chi^2_4 acceptance exactly.
    """
    from scipy.stats import chi2
    D = data
    sub = np.arange(min(D.N, n_max))
    fake = object.__new__(XcovData)
    fake.N = len(sub)
    fake.X, fake.E, fake.z = D.mu2[sub], D.E[sub], D.z[sub]
    fake.model, fake.mu2, fake.c = D.mu2[sub], D.mu2[sub], np.zeros((len(sub), 4))
    fake.G, fake.k4, fake.ref = D.G[sub], D.k4[sub], D.ref[sub]
    fake.mu1, fake.t1 = D.mu2[sub][np.arange(len(sub)), D.ref[sub]], np.full(len(sub), 1e6)
    worst = 0.0
    for a in range(4):
        for b in range(a + 1, 4):
            like = PairLikelihood(fake, COLOURS[a], COLOURS[b], RSet(np.array([0.0, 1.0])),
                                  {c: (lambda z: 0 * z) for c in COLOURS}, n_quad=n_quad)
            G = D.G[sub]
            S00, S11, S01 = G[:, a, a], G[:, b, b], G[:, a, b]
            det = S00 * S11 - S01 ** 2
            q = (S11[:, None] * like.Vi ** 2 - 2 * S01[:, None] * like.Vi * like.Vj
                 + S00[:, None] * like.Vj ** 2) / det[:, None]
            P = np.sum(like.W * np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))[:, None], 1)
            worst = max(worst, float(np.max(np.abs(P - chi2.cdf(D.k4[sub], 4)))))
    return worst
