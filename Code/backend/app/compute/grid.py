from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import math

from .transforms import solve_affine5_like, solve_affine6, solve_helmert4


def build_distortion_grid(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a distortion grid similar to MapAnalyst.

    The payload is expected to contain:
      {
        "links": [
          {
            "name": str,
            "new": {"lat": float, "lng": float} | {"x": float, "y": float},
            "old": {...},
            "vx": float,   # optional residual in x-direction (same unit as new.x)
            "vy": float,   # optional residual in y-direction (same unit as new.y)
            "predicted": { ... },  # optional predicted position (used to derive residuals)
          }, ...
        ],
        "options": {
          "mesh_size": float,
          "mesh_unit": "m" | "deg",
          "extent": "rectangular" | "aroundPoints" | "customPolygon",
          "smoothness": float (0..100),
          "offset_x": float,
          "offset_y": float,
          "show_undistorted": bool,
          "uncertainty_quantile": float (0..100),
          "exaggeration": float,
          "polygon": [{"lat": float, "lng": float}, ...],
          "transform_kind": "helmert4" | "affine6" | "affine5"
        }
      }
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object (dict)")

    links = payload.get("links")
    if not links:
        raise ValueError("At least one link is required to build a grid")
    if len(links) < 3:
        raise ValueError("At least three links are required for the distortion grid")

    options = payload.get("options") or {}
    mesh_size = _safe_float(options.get("mesh_size") or options.get("mesh_size_m") or 5000.0)
    mesh_unit_raw = options.get("mesh_unit") or options.get("mesh_unit_kind") or "m"
    mesh_unit = str(mesh_unit_raw).lower()
    extent_mode = (options.get("extent") or "rectangular").lower()
    smoothness = _clamp(_safe_float(options.get("smoothness", 50.0)), 0.0, 100.0)
    show_undistorted = bool(options.get("show_undistorted") or options.get("show_undistorted_grid"))
    uncertainty_q = _clamp(_safe_float(options.get("uncertainty_quantile", options.get("uncertainty_quantile_pct", 75.0))), 0.0, 100.0)
    exaggeration = _safe_float(options.get("exaggeration", 1.0))
    transform_kind = str(options.get("transform_kind") or options.get("transform") or "helmert4").lower()

    offsets = options.get("offsets")
    if isinstance(offsets, Sequence) and len(offsets) >= 2:
        offset_x_raw = _safe_float(offsets[0])
        offset_y_raw = _safe_float(offsets[1])
    else:
        offset_x_raw = _safe_float(options.get("offset_x", 0.0))
        offset_y_raw = _safe_float(options.get("offset_y", 0.0))

    control_points: List[Tuple[float, float]] = []  # lat, lng
    source_points: List[Tuple[float, float]] = []   # lat, lng (old map)
    residuals_latlng: List[Tuple[float, float] | None] = []

    for ln in links:
        if not isinstance(ln, dict):
            continue
        coord_new = ln.get("new") or ln.get("coord") or {}
        lat_new, lng_new = _extract_latlng(coord_new)
        control_points.append((lat_new, lng_new))

        coord_old = ln.get("old") or {}
        if coord_old:
            lat_old, lng_old = _extract_latlng(coord_old)
            source_points.append((lat_old, lng_old))
        else:
            source_points.append(None)  # type: ignore[arg-type]

        vx = ln.get("vx")
        vy = ln.get("vy")
        if vx is None or vy is None:
            pred = ln.get("predicted") or ln.get("predicted_new") or {}
            if pred:
                plat, plng = _extract_latlng(pred)
                vx = lng_new - plng
                vy = lat_new - plat
        if vx is None or vy is None:
            residuals_latlng.append(None)
        else:
            residuals_latlng.append((_safe_float(vy), _safe_float(vx)))

    need_residuals = any(r is None for r in residuals_latlng)
    if need_residuals:
        if any(sp is None for sp in source_points):
            raise ValueError("Residuals missing and some links have no old-map coordinates")
        residuals_latlng = _derive_residuals(
            [sp for sp in source_points if sp is not None],
            control_points,
            transform_kind,
        )

    projection = _LocalProjection(control_points)
    ctrl_xy = [projection.to_local(lat, lng) for lat, lng in control_points]
    res_xy = [projection.residual_to_local(dlat, dlng) for dlat, dlng in residuals_latlng]

    step_x, step_y = _resolve_mesh_steps(mesh_size, mesh_unit, projection)
    off_x = projection.offset_to_local(offset_x_raw, axis="x", unit=mesh_unit)
    off_y = projection.offset_to_local(offset_y_raw, axis="y", unit=mesh_unit)

    extent_polygon = _build_extent_polygon(extent_mode, control_points, projection, options)
    if not extent_polygon:
        raise ValueError("Could not derive extent polygon for distortion grid")
    xs = [p[0] for p in extent_polygon]
    ys = [p[1] for p in extent_polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    vertical_positions, step_x = _grid_positions_with_limits(
        min_x, max_x, step_x, off_x
    )
    horizontal_positions, step_y = _grid_positions_with_limits(
        min_y, max_y, step_y, off_y
    )
    if not vertical_positions or not horizontal_positions:
        raise ValueError("Grid extent too small for chosen mesh size")
    if len(vertical_positions) * len(horizontal_positions) > 40000:
        raise ValueError("Grid is too dense. Increase mesh size or reduce extent.")

    power = 1.0 + 2.0 * (smoothness / 100.0)  # 1.0 .. 3.0
    interpolator = _IDW(ctrl_xy, res_xy, power=power, exaggeration=exaggeration)

    nn_dists = _nearest_neighbor_distances(ctrl_xy)
    threshold = _quantile(nn_dists, uncertainty_q / 100.0) if nn_dists else 0.0

    def is_uncertain(xy: Tuple[float, float]) -> bool:
        if not nn_dists:
            return False
        return _min_distance(xy, ctrl_xy) > threshold

    distorted_lines: List[Dict[str, Any]] = []
    undistorted_lines: List[Dict[str, Any]] = []

    for x in vertical_positions:
        seq: List[Dict[str, Any]] = []
        seq_undist: List[Dict[str, Any]] = []
        for y in horizontal_positions:
            if not _point_in_polygon((x, y), extent_polygon):
                if seq:
                    distorted_lines.append({"orientation": "vertical", "points": seq})
                    if show_undistorted:
                        undistorted_lines.append({"orientation": "vertical", "points": seq_undist})
                    seq, seq_undist = [], []
                continue
            dx, dy = interpolator.at((x, y))
            gx, gy = x + dx, y + dy
            lat, lng = projection.to_geo(gx, gy)
            lat0, lng0 = projection.to_geo(x, y)
            seq.append({"lat": lat, "lng": lng, "uncertain": is_uncertain((x, y))})
            if show_undistorted:
                seq_undist.append({"lat": lat0, "lng": lng0, "uncertain": False})
        if seq:
            distorted_lines.append({"orientation": "vertical", "points": seq})
            if show_undistorted:
                undistorted_lines.append({"orientation": "vertical", "points": seq_undist})

    for y in horizontal_positions:
        seq: List[Dict[str, Any]] = []
        seq_undist: List[Dict[str, Any]] = []
        for x in vertical_positions:
            if not _point_in_polygon((x, y), extent_polygon):
                if seq:
                    distorted_lines.append({"orientation": "horizontal", "points": seq})
                    if show_undistorted:
                        undistorted_lines.append({"orientation": "horizontal", "points": seq_undist})
                    seq, seq_undist = [], []
                continue
            dx, dy = interpolator.at((x, y))
            gx, gy = x + dx, y + dy
            lat, lng = projection.to_geo(gx, gy)
            lat0, lng0 = projection.to_geo(x, y)
            seq.append({"lat": lat, "lng": lng, "uncertain": is_uncertain((x, y))})
            if show_undistorted:
                seq_undist.append({"lat": lat0, "lng": lng0, "uncertain": False})
        if seq:
            distorted_lines.append({"orientation": "horizontal", "points": seq})
            if show_undistorted:
                undistorted_lines.append({"orientation": "horizontal", "points": seq_undist})

    return {
        "meta": {
            "extent": extent_mode,
            "mesh_step_x_m": step_x,
            "mesh_step_y_m": step_y,
            "offset_x_m": off_x,
            "offset_y_m": off_y,
            "node_count": len(vertical_positions) * len(horizontal_positions),
            "line_count": len(distorted_lines),
            "uncertainty_threshold_m": threshold,
            "transform_kind": transform_kind,
        },
        "distorted": distorted_lines,
        "undistorted": undistorted_lines if show_undistorted else None,
    }


# ---------------------------------------------------------------------------
# Helpers


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _extract_latlng(coord: Dict[str, Any]) -> Tuple[float, float]:
    if not isinstance(coord, dict):
        raise ValueError("Coordinate must be an object")
    if "lat" in coord and "lng" in coord:
        return _safe_float(coord["lat"]), _safe_float(coord["lng"])
    if "x" in coord and "y" in coord:
        # Frontend uses x=lng, y=lat internally
        return _safe_float(coord["y"]), _safe_float(coord["x"])
    raise ValueError("Coordinate must have lat/lng or x/y fields")


def _derive_residuals(old_pts: Sequence[Tuple[float, float]],
                      new_pts: Sequence[Tuple[float, float]],
                      kind: str) -> List[Tuple[float, float]]:
    src_x = [lng for _, lng in old_pts]
    src_y = [lat for lat, _ in old_pts]
    trg_x = [lng for _, lng in new_pts]
    trg_y = [lat for lat, _ in new_pts]

    kind = kind.lower()
    if kind == "affine6":
        solver = solve_affine6
    elif kind in {"affine5", "affine5_like"}:
        solver = solve_affine5_like
    else:
        solver = solve_helmert4

    sol = solver(src_x, src_y, trg_x, trg_y)
    res_x = sol["residuals_x"]
    res_y = sol["residuals_y"]
    residuals = []
    for rx, ry in zip(res_x, res_y):
        # Residuals from sol are predicted - observed; invert to obtain observed - predicted
        residuals.append((-ry, -rx))
    return residuals


class _LocalProjection:
    """Simple equirectangular projection around the mean latitude/longitude."""

    def __init__(self, coords: Sequence[Tuple[float, float]]):
        if not coords:
            raise ValueError("No coordinates supplied")
        self.lat0 = sum(lat for lat, _ in coords) / len(coords)
        self.lng0 = sum(lng for _, lng in coords) / len(coords)
        self._m_per_deg_lat = 111_320.0
        cos_lat = math.cos(math.radians(self.lat0))
        self._m_per_deg_lng = max(self._m_per_deg_lat * max(cos_lat, 1e-6), 1.0)

    def to_local(self, lat: float, lng: float) -> Tuple[float, float]:
        x = (lng - self.lng0) * self._m_per_deg_lng
        y = (lat - self.lat0) * self._m_per_deg_lat
        return (x, y)

    def to_geo(self, x: float, y: float) -> Tuple[float, float]:
        lat = self.lat0 + y / self._m_per_deg_lat
        lng = self.lng0 + x / self._m_per_deg_lng
        return (lat, lng)

    def residual_to_local(self, dlat: float, dlng: float) -> Tuple[float, float]:
        return (
            dlng * self._m_per_deg_lng,
            dlat * self._m_per_deg_lat,
        )

    def offset_to_local(self, value: float, *, axis: str, unit: str) -> float:
        deg_aliases = {"deg", "degree", "degrees", "°"}
        if unit in deg_aliases:
            if axis == "x":
                return value * self._m_per_deg_lng
            return value * self._m_per_deg_lat
        return value


class _IDW:
    def __init__(self, points: Sequence[Tuple[float, float]],
                 vectors: Sequence[Tuple[float, float]], *,
                 power: float = 2.0, exaggeration: float = 1.0) -> None:
        if len(points) != len(vectors):
            raise ValueError("Points and vectors length mismatch")
        self.points = list(points)
        self.vectors = list(vectors)
        self.power = max(0.1, float(power))
        self.exaggeration = float(exaggeration)

    def at(self, xy: Tuple[float, float]) -> Tuple[float, float]:
        x, y = xy
        num_x = num_y = 0.0
        denom = 0.0
        for (px, py), (vx, vy) in zip(self.points, self.vectors):
            dx = x - px
            dy = y - py
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                return vx * self.exaggeration, vy * self.exaggeration
            weight = 1.0 / (dist ** self.power)
            denom += weight
            num_x += vx * weight
            num_y += vy * weight
        if denom == 0.0:
            return (0.0, 0.0)
        return (num_x / denom) * self.exaggeration, (num_y / denom) * self.exaggeration


def _nearest_neighbor_distances(points: Sequence[Tuple[float, float]]) -> List[float]:
    out: List[float] = []
    for i, (x, y) in enumerate(points):
        best = math.inf
        for j, (xx, yy) in enumerate(points):
            if i == j:
                continue
            d = math.hypot(x - xx, y - yy)
            if d < best:
                best = d
        if math.isfinite(best):
            out.append(best)
    return out


def _min_distance(pt: Tuple[float, float], points: Sequence[Tuple[float, float]]) -> float:
    best = math.inf
    x, y = pt
    for px, py in points:
        d = math.hypot(x - px, y - py)
        if d < best:
            best = d
    return best


def _quantile(data: Sequence[float], q: float) -> float:
    if not data:
        return 0.0
    q = _clamp(q, 0.0, 1.0)
    sorted_data = sorted(data)
    if q <= 0:
        return sorted_data[0]
    if q >= 1:
        return sorted_data[-1]
    pos = (len(sorted_data) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_data[lower]
    fraction = pos - lower
    return sorted_data[lower] * (1 - fraction) + sorted_data[upper] * fraction


def _resolve_mesh_steps(mesh_size: float, unit: str, projection: _LocalProjection) -> Tuple[float, float]:
    unit_norm = unit.lower()
    if unit_norm not in {"m", "meter", "meters", "deg", "degree", "degrees", "°"}:
        unit_norm = "m"
    if unit_norm.startswith("deg") or unit_norm == "°":
        step_x = mesh_size * projection._m_per_deg_lng
        step_y = mesh_size * projection._m_per_deg_lat
    else:
        step_x = step_y = mesh_size
    if step_x <= 0 or step_y <= 0:
        raise ValueError("Mesh size must be > 0")
    return step_x, step_y


def _build_extent_polygon(mode: str, control_points: Sequence[Tuple[float, float]],
                          projection: _LocalProjection, options: Dict[str, Any]) -> List[Tuple[float, float]]:
    mode = mode.lower()
    if mode == "custompolygon":
        raw = options.get("polygon") or []
        poly: List[Tuple[float, float]] = []
        for item in raw:
            lat, lng = _extract_latlng(item)
            poly.append(projection.to_local(lat, lng))
        if len(poly) < 3:
            raise ValueError("Custom polygon requires at least 3 vertices")
        return poly

    pts_local = [projection.to_local(lat, lng) for lat, lng in control_points]
    if mode == "aroundpoints" and len(pts_local) >= 3:
        return _convex_hull(pts_local)

    xs = [x for x, _ in pts_local]
    ys = [y for _, y in pts_local]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    expand = max(max_x - min_x, max_y - min_y) * 0.05
    if not math.isfinite(expand):
        expand = 0.0
    min_x -= expand
    max_x += expand
    min_y -= expand
    max_y += expand
    return [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]


def _convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return list(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull


def _grid_positions(min_val: float, max_val: float, step: float, offset: float) -> List[float]:
    if step <= 0:
        raise ValueError("Grid step must be > 0")
    offset = ((offset % step) + step) % step
    start_idx = math.floor((min_val - offset) / step) - 1
    positions: List[float] = []
    idx = start_idx
    while True:
        pos = offset + idx * step
        if pos > max_val + 1e-6:
            break
        if pos >= min_val - 1e-6:
            positions.append(pos)
        idx += 1
    return positions


def _grid_positions_with_limits(min_val: float, max_val: float, step: float, offset: float,
                                _min_lines: int = 2, max_lines: int = 200) -> Tuple[List[float], float]:
    """Return grid positions without shrinking the requested mesh size.

    We keep the caller's step (mesh size) to preserve the expected spacing and only
    relax it when the grid would become excessively dense.
    """
    span = max_val - min_val
    if not math.isfinite(span) or span <= 0:
        return [min_val, max_val], span if span > 0 else step

    base_step = float(step)
    if base_step <= 0:
        raise ValueError("Grid step must be > 0")

    positions = _grid_positions(min_val, max_val, base_step, offset)
    if not positions:
        return positions, base_step

    count = len(positions)
    if count > max_lines:
        factor = max(1, math.ceil(count / max_lines))
        adjusted_step = base_step * factor
        positions = _grid_positions(min_val, max_val, adjusted_step, offset)
        return positions, adjusted_step

    return positions, base_step


def _point_in_polygon(pt: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    x, y = pt
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Check if point is on edge (within tolerance)
        if _on_segment((x1, y1), (x2, y2), (x, y)):
            return True
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        if intersects:
            inside = not inside
    return inside


def _on_segment(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float], tol: float = 1e-6) -> bool:
    (x1, y1), (x2, y2), (x, y) = a, b, p
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > tol:
        return False
    dot = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    if dot < -tol:
        return False
    squared_len = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if dot > squared_len + tol:
        return False
    return True
