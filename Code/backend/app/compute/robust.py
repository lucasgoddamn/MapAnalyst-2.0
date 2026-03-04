# backend/app/compute/robust.py
from typing import Dict, List
import math

try:
    import numpy as np
except Exception:
    np = None


def _require_numpy():
    if np is None:
        raise RuntimeError(
            "NumPy is required. Install with `pip install numpy`."
        )


def _helmert_weighted(X: "np.ndarray", Y: "np.ndarray", w: "np.ndarray"):
    """
    Weighted similarity (Umeyama-like) solving Y ≈ s R X + t
    X,Y: 2xN, w: Nx weights (>=0)
    Returns s, R(2x2), t(2,)
    """
    _require_numpy()
    w = w.reshape(-1, 1)   # N x 1
    Wsum = float(w.sum())
    if Wsum <= 0:
        raise ValueError("All weights are zero.")

    # Weighted centroids
    mu_x = (X @ w) / Wsum
    mu_y = (Y @ w) / Wsum
    Xc = X - mu_x
    Yc = Y - mu_y

    # Weighted covariance
    Sigma = (Yc * w.T) @ Xc.T / Wsum
    U, D, Vt = np.linalg.svd(Sigma)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        S = np.eye(2)
        S[1, 1] = -1
        R = U @ S @ Vt
        D[1] = -D[1]

    var_x = float(((Xc**2) * w.T).sum() / Wsum)
    if var_x <= 0:
        raise ValueError("Degenerate source points (zero variance).")

    s = float(D.sum() / var_x)
    t = (mu_y - s * R @ mu_x).reshape(2)
    return s, R, t


def _robust_weights(norm_v: "np.ndarray", kind: str, params: Dict) -> "np.ndarray":
    """
    kind: 'huber' | 'v' (Tukey biweight) | 'hampel'
    params: dict of dicts from request
    """
    _require_numpy()
    r = norm_v.copy()
    eps = 1e-12

    if kind == "huber":
        k = float(params.get("huber", {}).get("k", 1.5))
        w = np.ones_like(r)
        mask = r > k
        w[mask] = k / (r[mask] + eps)
        return w

    if kind == "v":
        # Tukey's biweight with tuning k; e is not used directly here, but kept for parity
        k = float(params.get("v", {}).get("k", 1.5))
        w = np.zeros_like(r)
        mask = r < k
        u = r[mask] / k
        w[mask] = (1 - u**2) ** 2
        return w

    if kind == "hampel":
        # a < b < c
        HP = params.get("hampel", {})
        a = float(HP.get("a", 1.0))
        b = float(HP.get("b", 2.0))
        c = float(HP.get("c", 4.0))
        w = np.empty_like(r)
        w.fill(0.0)

        m1 = r <= a
        w[m1] = 1.0

        m2 = (r > a) & (r <= b)
        w[m2] = a / (r[m2] + eps)

        m3 = (r > b) & (r <= c)
        w[m3] = a * (c - r[m3]) / ((r[m3] + eps) * (c - b + eps))

        # r > c => 0 weight
        return w

    # default: no weighting
    return np.ones_like(r)


def solve_robust_helmert(
    src_x: List[float], src_y: List[float],
    trg_x: List[float], trg_y: List[float],
    kind: str = "huber",
    params: Dict = None,
    max_iter: int = 30,
    tol: float = 1e-8,
) -> Dict:
    """
    IRLS on similarity transform. Returns same structure as other solvers.
    """
    _require_numpy()
    X = np.vstack([np.asarray(src_x, dtype=float),
                   np.asarray(src_y, dtype=float)])  # 2xN
    Y = np.vstack([np.asarray(trg_x, dtype=float),
                   np.asarray(trg_y, dtype=float)])  # 2xN
    n = X.shape[1]
    if n < 2:
        raise ValueError("Robust Helmert needs at least 2 points.")
    if params is None:
        params = {}

    # init (all ones)
    w = np.ones((n,), dtype=float)

    s, R, t = 1.0, np.eye(2), np.zeros(2)
    last_obj = None

    for _ in range(max_iter):
        # weighted estimate
        s, R, t = _helmert_weighted(X, Y, w)
        E = (s * (R @ X) + t.reshape(2, 1)) - Y     # 2xN
        vnorm = np.sqrt((E**2).sum(axis=0))         # N
        # scale for robustification: MAD-like (or simple RMS)
        sigma = max(np.median(vnorm) / 0.6745, 1e-12)
        r = vnorm / sigma

        w = _robust_weights(r, kind, params)
        obj = float((w * (vnorm**2)).sum())
        if last_obj is not None and abs(last_obj - obj) < tol:
            break
        last_obj = obj

    # Final residuals
    E = (s * (R @ X) + t.reshape(2, 1)) - Y
    vx = E[0, :].tolist()
    vy = E[1, :].tolist()

    theta = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return {
        "params": {"scale": float(s), "rotation_deg": theta, "tx": float(t[0]), "ty": float(t[1])},
        "residuals_x": vx,
        "residuals_y": vy,
        "meta": {"model": "robust-helmert", "kind": kind},
    }
