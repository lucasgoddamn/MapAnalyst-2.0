from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np


def build_isolines(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute scale and rotation isolines based on linked points, approximating MapAnalyst.

    Expected payload structure:
      {
        "target": "old" | "new",
        "links": [
            {"old": {"x": ..., "y": ...}, "new": {"x": ..., "y": ...}},
            ...
        ],
        "options": {
            "radius": float,
            "scale": {"show": bool, "interval": float},
            "rotation": {"show": bool, "interval": float},
            "grid_samples": int (optional),
        }
      }

    Returns:
      {
        "target": "...",
        "radius": float,
        "bounds": {"min_x": ..., "max_x": ..., "min_y": ..., "max_y": ...},
        "scale": {"levels": [{"value": float, "lines": [[{"lat":..,"lng":..},...], ...]}, ...]},
        "rotation": {...}
      }
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object.")

    target = str(payload.get("target") or "old").lower()
    if target not in ("old", "new"):
        raise ValueError("target must be 'old' or 'new'.")

    links = payload.get("links") or []
    if len(links) < 3:
        raise ValueError("At least three links are required to build isolines.")

    options = payload.get("options") or {}
    radius = float(options.get("radius") or 10000.0)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be a positive number.")

    scale_opts = options.get("scale") or {}
    rot_opts = options.get("rotation") or {}
    scale_show = bool(scale_opts.get("show"))
    rot_show = bool(rot_opts.get("show"))
    scale_interval = _safe_positive(scale_opts.get("interval"), default=5000.0)
    rot_interval = _safe_positive(rot_opts.get("interval"), default=5.0)

    grid_samples = options.get("grid_samples")
    grid_samples = int(grid_samples) if grid_samples else None
    if grid_samples is not None:
        grid_samples = max(10, min(200, grid_samples))

    src_points: List[Tuple[float, float]] = []
    dst_points: List[Tuple[float, float]] = []
    for ln in links:
        if not isinstance(ln, dict):
            continue
        old = ln.get("old") or {}
        new = ln.get("new") or {}
        if not _has_xy(old) or not _has_xy(new):
            continue
        if target == "old":
            src_points.append((float(old["x"]), float(old["y"])))
            dst_points.append((float(new["x"]), float(new["y"])))
        else:
            src_points.append((float(new["x"]), float(new["y"])))
            dst_points.append((float(old["x"]), float(old["y"])))

    if len(src_points) < 3:
        raise ValueError("Not enough valid link coordinates to compute isolines.")

    xs = [pt[0] for pt in src_points]
    ys = [pt[1] for pt in src_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    if not (math.isfinite(span_x) and math.isfinite(span_y)):
        raise ValueError("Could not derive extent for isolines.")
    if span_x <= 0 or span_y <= 0:
        raise ValueError("Isoline extent too small (all points aligned).")

    padding = radius * 0.5
    grid_min_x = min_x - padding
    grid_max_x = max_x + padding
    grid_min_y = min_y - padding
    grid_max_y = max_y + padding

    if grid_samples:
        nx = ny = grid_samples
    else:
        nx = max(40, min(120, int(max(span_x, span_y) / max(radius * 0.3, 1e-6))))
        if nx % 2 == 1:
            nx += 1
        ny = nx

    xs_grid = np.linspace(grid_min_x, grid_max_x, nx)
    ys_grid = np.linspace(grid_min_y, grid_max_y, ny)

    scale_field = np.full((ny, nx), np.nan, dtype=float)
    rotation_field = np.full((ny, nx), np.nan, dtype=float)

    src_np = np.asarray(src_points, dtype=float)
    dst_np = np.asarray(dst_points, dtype=float)

    radius_sq = radius * radius
    for iy, gy in enumerate(ys_grid):
        for ix, gx in enumerate(xs_grid):
            # distances to all source points
            dists_sq = (src_np[:, 0] - gx) ** 2 + (src_np[:, 1] - gy) ** 2
            mask = dists_sq <= radius_sq
            if mask.sum() < 3:
                continue
            local_src = src_np[mask]
            local_dst = dst_np[mask]
            local_dists_sq = dists_sq[mask]

            weights = _kernel_weights(local_dists_sq, radius_sq)
            params = _weighted_affine(local_src, local_dst, weights)
            if params is None:
                continue
            a, b, c, d = params[0], params[1], params[2], params[3]
            A = np.array([[a, b], [c, d]], dtype=float)
            try:
                U, S, Vt = np.linalg.svd(A)
            except np.linalg.LinAlgError:
                continue

            # enforce proper rotation (determinant +1)
            if np.linalg.det(U @ Vt) < 0:
                Vt[-1, :] *= -1
            R = U @ Vt
            rot_rad = math.atan2(R[1, 0], R[0, 0])
            rotation_field[iy, ix] = math.degrees(rot_rad)

            # Geometric mean of singular values ~ isotropic scale
            s0 = abs(S[0])
            s1 = abs(S[1])
            scale_value = float(math.sqrt(max(s0 * s1, 0.0))) if s0 > 0 and s1 > 0 else float(np.mean(np.abs(S)))
            scale_field[iy, ix] = scale_value

    result: Dict[str, Any] = {
        "target": target,
        "radius": radius,
        "bounds": {
            "min_x": grid_min_x,
            "max_x": grid_max_x,
            "min_y": grid_min_y,
            "max_y": grid_max_y,
        },
        "scale": {"levels": []},
        "rotation": {"levels": []},
    }

    # compute isolines per field
    if scale_show:
        scale_levels = _build_levels(scale_field, scale_interval)
        scale_lines = []
        for level in scale_levels:
            segments = _marching_squares(xs_grid, ys_grid, scale_field, level)
            lines = _assemble_segments(segments)
            geo_lines = [_to_geo_line(line) for line in lines if len(line) >= 2]
            if geo_lines:
                scale_lines.append({"value": level, "lines": geo_lines})
        result["scale"]["levels"] = scale_lines

    if rot_show:
        rot_levels = _build_levels(rotation_field, rot_interval)
        rot_lines = []
        for level in rot_levels:
            segments = _marching_squares(xs_grid, ys_grid, rotation_field, level)
            lines = _assemble_segments(segments)
            geo_lines = [_to_geo_line(line) for line in lines if len(line) >= 2]
            if geo_lines:
                rot_lines.append({"value": level, "lines": geo_lines})
        result["rotation"]["levels"] = rot_lines

    return result


def _has_xy(obj: Any) -> bool:
    try:
        float(obj["x"])
        float(obj["y"])
        return True
    except Exception:
        return False


def _safe_positive(value: Any, default: float) -> float:
    try:
        num = float(value)
        if math.isfinite(num) and num > 0:
            return num
    except Exception:
        pass
    return default


def _kernel_weights(dists_sq: np.ndarray, radius_sq: float) -> np.ndarray:
    """
    Compact, smooth kernel (quartic) that vanishes at radius and peaks at center.
    """
    if radius_sq <= 0:
        return np.ones_like(dists_sq)
    normalized = np.clip(dists_sq / radius_sq, 0.0, 1.0)
    weights = (1.0 - normalized) ** 2
    weights_sum = weights.sum()
    if weights_sum <= 0:
        return np.ones_like(weights)
    return weights / weights_sum


def _weighted_affine(
    src: np.ndarray, dst: np.ndarray, weights: np.ndarray
) -> Optional[np.ndarray]:
    """
    Weighted least-squares affine fit mapping src -> dst.
    Returns parameters [a, b, c, d, tx, ty] or None if solve fails.
    """
    n = src.shape[0]
    if n < 3:
        return None
    x = src[:, 0]
    y = src[:, 1]
    Xp = dst[:, 0]
    Yp = dst[:, 1]
    w = weights.reshape(-1, 1)
    sqrt_w = np.sqrt(weights).reshape(-1, 1)

    A = np.zeros((2 * n, 6), dtype=float)
    A[0:n, 0] = x
    A[0:n, 1] = y
    A[0:n, 4] = 1.0
    A[n:2 * n, 2] = x
    A[n:2 * n, 3] = y
    A[n:2 * n, 5] = 1.0

    # apply weights
    A[0:n] *= sqrt_w
    A[n:2 * n] *= sqrt_w

    L = np.concatenate([Xp, Yp], axis=0)
    L[0:n] *= sqrt_w.flatten()
    L[n:2 * n] *= sqrt_w.flatten()

    try:
        params, *_ = np.linalg.lstsq(A, L, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return params


def _build_levels(field: np.ndarray, interval: float) -> List[float]:
    values = field[~np.isnan(field)]
    if values.size == 0 or interval <= 0:
        return []
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if min_val == max_val:
        return [min_val]
    start = math.floor(min_val / interval) * interval
    end = math.ceil(max_val / interval) * interval
    levels = []
    level = start
    # add small epsilon to prevent infinite loops due to floating precision
    while level <= end + interval * 0.5:
        levels.append(round(level, 12))
        level += interval
    return levels


def _marching_squares(
    xs: np.ndarray, ys: np.ndarray, field: np.ndarray, level: float
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return list of contour segments for a single level."""
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    nx = xs.size
    ny = ys.size
    for j in range(ny - 1):
        y0 = ys[j]
        y1 = ys[j + 1]
        for i in range(nx - 1):
            v0 = field[j, i]
            v1 = field[j, i + 1]
            v2 = field[j + 1, i + 1]
            v3 = field[j + 1, i]
            if (
                math.isnan(v0)
                or math.isnan(v1)
                or math.isnan(v2)
                or math.isnan(v3)
            ):
                continue
            idx = 0
            if v0 >= level:
                idx |= 1
            if v1 >= level:
                idx |= 2
            if v2 >= level:
                idx |= 4
            if v3 >= level:
                idx |= 8
            if idx == 0 or idx == 15:
                continue
            x0 = xs[i]
            x1 = xs[i + 1]
            case = _MS_CASES[idx]
            edge_points = []
            for edge in case:
                p1, p2 = _edge_points(edge, level, x0, x1, y0, y1, v0, v1, v2, v3)
                if p1 is not None and p2 is not None:
                    edge_points.append((p1, p2))
            for seg in edge_points:
                segments.append(seg)
    return segments


# Marching squares cases referencing edges 0(bottom),1(right),2(top),3(left)
_MS_CASES = {
    0: [],
    1: [(3, 0)],
    2: [(0, 1)],
    3: [(3, 1)],
    4: [(1, 2)],
    5: [(3, 2), (0, 1)],
    6: [(0, 2)],
    7: [(3, 2)],
    8: [(2, 3)],
    9: [(0, 2)],
    10: [(0, 1), (2, 3)],
    11: [(1, 2)],
    12: [(1, 3)],
    13: [(0, 1)],
    14: [(3, 0)],
    15: [],
}


def _edge_points(
    edge: Tuple[int, int],
    level: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    v0: float,
    v1: float,
    v2: float,
    v3: float,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    e_from, e_to = edge
    p_from = _interpolate_edge(e_from, level, x0, x1, y0, y1, v0, v1, v2, v3)
    p_to = _interpolate_edge(e_to, level, x0, x1, y0, y1, v0, v1, v2, v3)
    return p_from, p_to


def _interpolate_edge(
    edge: int,
    level: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    v0: float,
    v1: float,
    v2: float,
    v3: float,
) -> Optional[Tuple[float, float]]:
    # helper to choose vertices
    if edge == 0:  # bottom v0->v1
        return _lerp((x0, y0), (x1, y0), v0, v1, level)
    if edge == 1:  # right v1->v2
        return _lerp((x1, y0), (x1, y1), v1, v2, level)
    if edge == 2:  # top v2->v3
        return _lerp((x1, y1), (x0, y1), v2, v3, level)
    if edge == 3:  # left v3->v0
        return _lerp((x0, y1), (x0, y0), v3, v0, level)
    return None


def _lerp(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    v1: float,
    v2: float,
    level: float,
) -> Optional[Tuple[float, float]]:
    if v1 == v2:
        t = 0.5
    else:
        t = (level - v1) / (v2 - v1)
    t = max(0.0, min(1.0, t))
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))


def _assemble_segments(
    segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    tol: float = 1e-6,
) -> List[List[Tuple[float, float]]]:
    lines: List[List[Tuple[float, float]]] = []
    remaining = list(segments)
    while remaining:
        start, end = remaining.pop()
        line = [start, end]
        extended = True
        while extended:
            extended = False
            for idx, seg in enumerate(remaining):
                s, e = seg
                if _close_points(line[0], e, tol):
                    line.insert(0, s)
                elif _close_points(line[0], s, tol):
                    line.insert(0, e)
                elif _close_points(line[-1], s, tol):
                    line.append(e)
                elif _close_points(line[-1], e, tol):
                    line.append(s)
                else:
                    continue
                remaining.pop(idx)
                extended = True
                break
        lines.append(line)
    return lines


def _close_points(
    p1: Tuple[float, float], p2: Tuple[float, float], tol: float
) -> bool:
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 <= tol * tol


def _to_geo_line(line: Sequence[Tuple[float, float]]) -> List[Dict[str, float]]:
    return [{"lat": pt[1], "lng": pt[0]} for pt in line]
