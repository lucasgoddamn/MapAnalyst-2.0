# backend/app/compute/transforms.py
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


def _as_xy(src_x: List[float], src_y: List[float]) -> "np.ndarray":
    _require_numpy()
    X = np.vstack([np.asarray(src_x, dtype=float),
                   np.asarray(src_y, dtype=float)])  # 2xN
    if X.shape[1] < 2:
        raise ValueError("At least 2 points required.")
    return X


def _affine_design(src_x, src_y):
    """
    Build LS design matrix for affine (6 params: a,b,c,d,tx,ty)
    Y = [a b; c d] * X + [tx; ty]
    """
    _require_numpy()
    x = np.asarray(src_x, dtype=float)
    y = np.asarray(src_y, dtype=float)
    n = x.size

    # For X' (target x):  [x y 0 0 1 0] [a b c d tx ty]^T
    # For Y' (target y):  [0 0 x y 0 1] [a b c d tx ty]^T
    A = np.zeros((2*n, 6), dtype=float)
    A[0:n, 0] = x
    A[0:n, 1] = y
    A[0:n, 4] = 1.0          # tx
    A[n:2*n, 2] = x
    A[n:2*n, 3] = y
    A[n:2*n, 5] = 1.0        # ty
    return A


def _solve_lstsq(A, L):
    _require_numpy()
    p, *_ = np.linalg.lstsq(A, L, rcond=None)
    return p


def solve_helmert4(
    src_x: List[float], src_y: List[float],
    trg_x: List[float], trg_y: List[float]
) -> Dict:
    """
    2D Similarity (scale s, rotation theta, translation tx, ty)
    Returns:
      {
        "params": {"scale": s, "rotation_deg": theta_deg, "tx": tx, "ty": ty},
        "residuals_x": vx, "residuals_y": vy
      }
    """
    _require_numpy()
    X = _as_xy(src_x, src_y)  # 2xN
    Y = _as_xy(trg_x, trg_y)  # 2xN
    n = X.shape[1]

    # Center
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)
    Xc = X - mu_x
    Yc = Y - mu_y

    # Cross-covariance
    Sigma = (Yc @ Xc.T) / n
    U, D, Vt = np.linalg.svd(Sigma)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        S = np.eye(2)
        S[1, 1] = -1
        R = U @ S @ Vt
        D[1] = -D[1]

    var_x = (Xc * Xc).sum() / n
    if var_x <= 0:
        raise ValueError("Degenerate source points (zero variance).")

    s = float(D.sum() / var_x)
    t = (mu_y - s * R @ mu_x).reshape(2)

    A = s * R
    est = A @ X + t.reshape(2, 1)
    V = est - Y
    vx = V[0, :].copy()
    vy = V[1, :].copy()

    theta = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return {
        "params": {"scale": s, "rotation_deg": theta, "tx": float(t[0]), "ty": float(t[1])},
        "residuals_x": vx.tolist(),
        "residuals_y": vy.tolist(),
        "meta": {"model": "helmert4"},
    }


def solve_affine6(
    src_x: List[float], src_y: List[float],
    trg_x: List[float], trg_y: List[float]
) -> Dict:
    """
    Full affine: X' = a*x + b*y + tx
                 Y' = c*x + d*y + ty
    6 parameters
    """
    _require_numpy()
    x = np.asarray(src_x, dtype=float)
    y = np.asarray(src_y, dtype=float)
    Xp = np.asarray(trg_x, dtype=float)
    Yp = np.asarray(trg_y, dtype=float)
    n = x.size

    if n < 3:
        raise ValueError("Affine6 needs at least 3 points.")

    A = _affine_design(x, y)
    L = np.concatenate([Xp, Yp], axis=0)
    p = _solve_lstsq(A, L)  # [a,b,c,d,tx,ty]

    a, b, c, d, tx, ty = p.tolist()
    estX = a * x + b * y + tx
    estY = c * x + d * y + ty
    vx = (estX - Xp).tolist()
    vy = (estY - Yp).tolist()

    return {
        "params": {"a": a, "b": b, "c": c, "d": d, "tx": tx, "ty": ty},
        "residuals_x": vx,
        "residuals_y": vy,
        "meta": {"model": "affine6"},
    }


def solve_affine5_like(
    src_x: List[float], src_y: List[float],
    trg_x: List[float], trg_y: List[float]
) -> Dict:
    """
    A pragmatic 5-parameter linear model without shear:
        [a  -b] [x] + [tx]
        [b   d] [y]   [ty]
    Unknowns: a, b, d, tx, ty  (5 params)
    This allows anisotropic scale in axes aligned with rotation, but no shear.

    (Linear LS in those 5 unknowns.)
    """
    _require_numpy()
    x = np.asarray(src_x, dtype=float)
    y = np.asarray(src_y, dtype=float)
    Xp = np.asarray(trg_x, dtype=float)
    Yp = np.asarray(trg_y, dtype=float)
    n = x.size

    if n < 3:
        raise ValueError("Affine5-like needs at least 3 points.")

    # Build design for the two rows:
    # X' =  a*x  + (-b)*y + tx  -> [x, -y, 0, 1, 0]
    # Y' =  b*x  +   d*y  + ty  -> [0,  x,  y, 0, 1]
    A = np.zeros((2*n, 5), dtype=float)
    A[0:n, 0] = x
    A[0:n, 1] = -y
    A[0:n, 3] = 1.0  # tx
    A[n:2*n, 1] = x
    A[n:2*n, 2] = y
    A[n:2*n, 4] = 1.0  # ty

    L = np.concatenate([Xp, Yp], axis=0)
    p = _solve_lstsq(A, L)  # [a, b, d, tx, ty]
    a, b, d, tx, ty = p.tolist()

    estX = a * x + (-b) * y + tx
    estY = b * x + d * y + ty
    vx = (estX - Xp).tolist()
    vy = (estY - Yp).tolist()

    # Derive an equivalent scale/rotation-ish description (optional)
    # Not exact if a != d, but useful meta
    rot_deg = math.degrees(math.atan2(b, a))
    meta = {"approx_rotation_deg": rot_deg}

    return {
        "params": {"a": a, "b": b, "d": d, "tx": tx, "ty": ty},
        "residuals_x": vx,
        "residuals_y": vy,
        "meta": {"model": "affine5_like", **meta},
    }
