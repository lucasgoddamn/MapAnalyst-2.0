from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

import numpy as np
import requests
try:  # pragma: no cover - heavy optional dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
from fastapi import HTTPException
from PIL import Image

try:  # pragma: no cover - heavy optional dependency
    from kornia.feature import LoFTR  # type: ignore
except Exception:  # pragma: no cover
    LoFTR = None  # type: ignore

try:  # pragma: no cover - OCR is optional
    import pytesseract  # type: ignore
    from pytesseract import Output as TesseractOutput  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore

from .storage import OSM_CACHE_DIR, resolve_storage_path

TILE_SIZE = 256
MAX_REGION_DIM = 2048
LOFTR_MAX_DIM = 1600
CONFIDENCE_THRESHOLD = 0.6
RANSAC_THRESHOLD = 28.0
SEED_RANSAC_THRESHOLD = 8.0
SEED_CONSISTENCY_PX = 45.0
RANSAC_FALLBACK_TOPK = 120
USER_AGENT = "MapAnalyst/0.1 (+https://mapanalysttwo.app)"
SUGGESTION_LIMIT = 80


@dataclass
class Region:
    image: np.ndarray
    origin: Tuple[float, float]
    to_map: Callable[[Tuple[float, float]], Tuple[float, float]]
    from_map: Callable[[Tuple[float, float]], Optional[Tuple[float, float]]]
    mask: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)


def clamp_lat(value: float) -> float:
    return max(min(value, 85.05112878), -85.05112878)


def latlon_to_global_pixel(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    lat = clamp_lat(lat)
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    scale = TILE_SIZE * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * scale
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * scale
    return x, y


def global_pixel_to_latlon(px: float, py: float, zoom: int) -> Tuple[float, float]:
    scale = TILE_SIZE * (2 ** zoom)
    lon = px / scale * 360.0 - 180.0
    lat_rad = math.pi - 2.0 * math.pi * py / scale
    lat = math.degrees(math.atan(math.sinh(lat_rad)))
    return lat, lon


def _ensure_uint8_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if mask.dtype != np.uint8:
        return mask.astype(np.uint8)
    return mask


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _load_image_bgr(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as img:
            return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"Image file not found: {path}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to load image {path}: {exc}") from exc


def _polygon_bbox(points: Iterable[Dict[str, float]], width: int, height: int) -> Tuple[int, int, int, int]:
    xs: List[float] = []
    ys: List[float] = []
    for point in points:
        if "lng" in point and "lat" in point:
            xs.append(float(point["lng"]))
            ys.append(float(point["lat"]))
    if not xs or not ys:
        return 0, width, 0, height
    min_x = max(0, int(math.floor(min(xs))))
    max_x = min(width, int(math.ceil(max(xs))))
    min_y = max(0, int(math.floor(min(ys))))
    max_y = min(height, int(math.ceil(max(ys))))
    return min_x, max_x, min_y, max_y


def _create_polygon_mask(
    shape: Tuple[int, int],
    polygon: Iterable[Dict[str, float]],
    origin: Tuple[float, float],
    coordinate_adapter: Callable[[Dict[str, float]], Tuple[float, float]],
) -> Optional[np.ndarray]:
    pts: List[Tuple[float, float]] = []
    for pt in polygon:
        if "lat" in pt and "lng" in pt:
            local_x, local_y = coordinate_adapter(pt)
            pts.append((local_x - origin[0], local_y - origin[1]))
    if len(pts) < 3:
        return None
    poly = np.array([[pts]], dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, poly, 255)
    return mask


def _load_upload_region(
    source: Dict[str, Any],
    polygon: Optional[List[Dict[str, float]]],
    role: str,
) -> Region:
    stored = source.get("stored_as") or source.get("storage_path")
    if not stored:
        raise HTTPException(status_code=400, detail=f"No stored file reference for {role} map. Please upload it first.")
    try:
        path = resolve_storage_path(stored)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    image = _load_image_bgr(path)
    height, width = image.shape[:2]
    poly = polygon or []
    min_x, max_x, min_y, max_y = _polygon_bbox(poly, width, height)
    margin = 20
    min_x = max(0, min_x - margin)
    min_y = max(0, min_y - margin)
    max_x = min(width, max_x + margin)
    max_y = min(height, max_y + margin)
    if max_x <= min_x or max_y <= min_y:
        raise HTTPException(status_code=400, detail=f"{role.title()} map selection is empty.")
    cropped = image[min_y:max_y, min_x:max_x].copy()
    origin = (float(min_x), float(min_y))

    def to_map(pt: Tuple[float, float]) -> Tuple[float, float]:
        x, y = pt
        return float(y + origin[1]), float(x + origin[0])

    def from_map(latlng: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        lat, lng = latlng
        x = float(lng) - origin[0]
        y = float(lat) - origin[1]
        if x < 0 or y < 0 or x >= cropped.shape[1] or y >= cropped.shape[0]:
            return None
        return (x, y)

    mask = None
    if poly:
        mask = _create_polygon_mask(
            shape=cropped.shape[:2],
            polygon=poly,
            origin=origin,
            coordinate_adapter=lambda p: (float(p["lng"]), float(p["lat"])),
        )
    return Region(
        image=cropped,
        origin=origin,
        to_map=to_map,
        from_map=from_map,
        mask=_ensure_uint8_mask(mask),
        info={"role": role, "path": str(path)},
    )


def _fetch_tile(zoom: int, x: int, y: int, tileset: str) -> Image.Image:
    subdir = OSM_CACHE_DIR / str(zoom) / str(x)
    subdir.mkdir(parents=True, exist_ok=True)
    tile_path = subdir / f"{y}.png"
    if tile_path.exists():
        return Image.open(tile_path)
    subdomain = "abc"[(x + y) % 3]
    url = tileset.replace("{s}", subdomain).replace("{z}", str(zoom)).replace("{x}", str(x)).replace("{y}", str(y))
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to download OSM tile {zoom}/{x}/{y}: {exc}") from exc
    tile = Image.open(BytesIO(response.content)).convert("RGB")
    tile.save(tile_path)
    return tile


def _bounds_to_pixels(bounds: Dict[str, float], zoom: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    corners = [
        (bounds["north"], bounds["west"]),
        (bounds["north"], bounds["east"]),
        (bounds["south"], bounds["west"]),
        (bounds["south"], bounds["east"]),
    ]
    xs: List[float] = []
    ys: List[float] = []
    for lat, lon in corners:
        px, py = latlon_to_global_pixel(lat, lon, zoom)
        xs.append(px)
        ys.append(py)
    return (min(xs), min(ys)), (max(xs), max(ys))


def _fetch_osm_region(
    bounds: Dict[str, float],
    zoom: int,
    tileset: str,
) -> Tuple[Image.Image, float, float]:
    (min_px, min_py), (max_px, max_py) = _bounds_to_pixels(bounds, zoom)
    n_tiles = 2 ** zoom
    tile_x_min = max(0, int(math.floor(min_px / TILE_SIZE)))
    tile_x_max = min(n_tiles - 1, int(math.floor(max_px / TILE_SIZE)))
    tile_y_min = max(0, int(math.floor(min_py / TILE_SIZE)))
    tile_y_max = min(n_tiles - 1, int(math.floor(max_py / TILE_SIZE)))
    if tile_x_max < tile_x_min or tile_y_max < tile_y_min:
        raise HTTPException(status_code=400, detail="Invalid OSM bounds after normalization.")
    tile_count = (tile_x_max - tile_x_min + 1) * (tile_y_max - tile_y_min + 1)
    if tile_count > 120:
        raise HTTPException(status_code=400, detail="Requested OSM area is too large. Please zoom in further.")

    width = (tile_x_max - tile_x_min + 1) * TILE_SIZE
    height = (tile_y_max - tile_y_min + 1) * TILE_SIZE
    mosaic = Image.new("RGB", (width, height))
    for tx in range(tile_x_min, tile_x_max + 1):
        for ty in range(tile_y_min, tile_y_max + 1):
            tile = _fetch_tile(zoom, tx, ty, tileset)
            mosaic.paste(tile, ((tx - tile_x_min) * TILE_SIZE, (ty - tile_y_min) * TILE_SIZE))

    origin_global_x = tile_x_min * TILE_SIZE
    origin_global_y = tile_y_min * TILE_SIZE

    crop_left = max(0, int(math.floor(min_px - origin_global_x)))
    crop_top = max(0, int(math.floor(min_py - origin_global_y)))
    crop_right = min(width, int(math.ceil(max_px - origin_global_x)))
    crop_bottom = min(height, int(math.ceil(max_py - origin_global_y)))
    region = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))

    return region, origin_global_x + crop_left, origin_global_y + crop_top


def _load_osm_region(
    source: Dict[str, Any],
    view: Dict[str, Any],
    polygon: Optional[List[Dict[str, float]]],
) -> Region:
    tileset = source.get("tileset") or "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
    view_bounds = (view or {}).get("bounds") or {}
    zoom = int(round((view or {}).get("zoom") or 16))
    zoom = max(0, min(zoom, 19))

    bounds: Optional[Dict[str, float]] = None
    if polygon:
        lats = [float(pt["lat"]) for pt in polygon if "lat" in pt]
        lngs = [float(pt["lng"]) for pt in polygon if "lng" in pt]
        if lats and lngs:
            bounds = {
                "north": clamp_lat(max(lats)),
                "south": clamp_lat(min(lats)),
                "east": max(lngs),
                "west": min(lngs),
            }
    if not bounds:
        if not {"north", "south", "east", "west"} <= set(view_bounds.keys()):  # pragma: no cover - guard
            raise HTTPException(status_code=400, detail="Missing map bounds for OSM suggestions.")
        bounds = {
            "north": clamp_lat(float(view_bounds["north"])),
            "south": clamp_lat(float(view_bounds["south"])),
            "east": float(view_bounds["east"]),
            "west": float(view_bounds["west"]),
        }

    pad_lat = max(0.0005, (bounds["north"] - bounds["south"]) * 0.05 or 0.001)
    pad_lng = max(0.0005, (bounds["east"] - bounds["west"]) * 0.05 or 0.001)
    padded_bounds = {
        "north": clamp_lat(bounds["north"] + pad_lat),
        "south": clamp_lat(bounds["south"] - pad_lat),
        "east": bounds["east"] + pad_lng,
        "west": bounds["west"] - pad_lng,
    }

    region_img, origin_px, origin_py = _fetch_osm_region(padded_bounds, zoom, tileset)
    image = cv2.cvtColor(np.array(region_img), cv2.COLOR_RGB2BGR)

    def to_map(pt: Tuple[float, float]) -> Tuple[float, float]:
        x, y = pt
        lat, lng = global_pixel_to_latlon(origin_px + x, origin_py + y, zoom)
        return lat, lng

    def from_map(latlng: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        lat, lng = latlng
        px, py = latlon_to_global_pixel(lat, lng, zoom)
        local_x = px - origin_px
        local_y = py - origin_py
        if local_x < 0 or local_y < 0 or local_x >= image.shape[1] or local_y >= image.shape[0]:
            return None
        return (local_x, local_y)

    mask = None
    if polygon:
        mask = _create_polygon_mask(
            shape=image.shape[:2],
            polygon=polygon,
            origin=(origin_px, origin_py),
            coordinate_adapter=lambda p: latlon_to_global_pixel(float(p["lat"]), float(p["lng"]), zoom),
        )
    info = {"role": "new", "zoom": zoom, "bounds": padded_bounds}
    return Region(
        image=image,
        origin=(origin_px, origin_py),
        to_map=to_map,
        from_map=from_map,
        mask=_ensure_uint8_mask(mask),
        info=info,
    )


_DEVICE = torch.device("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu") if torch is not None else None
_LOFTR_MODEL: Optional[torch.nn.Module] = None


def _get_loftr_model() -> torch.nn.Module:
    global _LOFTR_MODEL
    if LoFTR is None:
        raise HTTPException(
            status_code=500,
            detail="Optional dependency 'kornia' with LoFTR is not installed. "
            "Install backend requirements to enable suggestions.",
        )
    if _LOFTR_MODEL is None:
        try:
            model = LoFTR(pretrained="outdoor")  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - download failure
            raise HTTPException(status_code=500, detail=f"Failed to load LoFTR weights: {exc}")
        model = model.to(_DEVICE)
        model.eval()
        _LOFTR_MODEL = model
    return _LOFTR_MODEL


def _prepare_loftr_input(region: Region) -> Tuple[torch.Tensor, float]:
    gray = cv2.cvtColor(region.image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    combined = cv2.addWeighted(gray, 0.65, edges, 0.35, 0)
    binary = cv2.adaptiveThreshold(
        combined,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    gray = cv2.bitwise_and(combined, binary)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    height, width = gray.shape[:2]
    scale = 1.0
    if max(height, width) > LOFTR_MAX_DIM:
        scale = LOFTR_MAX_DIM / max(height, width)
        new_size = (
            max(16, int(round(width * scale))),
            max(16, int(round(height * scale))),
        )
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
    return tensor.to(_DEVICE), scale


def _extract_seed_pairs(payload: Dict[str, Any], old_region: Region, new_region: Region) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    state = payload.get("state") or {}
    old_points = state.get("oldPoints") or []
    new_points = state.get("newPoints") or []
    seeds_old: List[np.ndarray] = []
    seeds_new: List[np.ndarray] = []
    for link in state.get("links") or []:
        oi = link.get("old_index")
        ni = link.get("new_index")
        if oi is None or ni is None or oi < 0 or ni < 0:
            continue
        if oi >= len(old_points) or ni >= len(new_points):
            continue
        old_map_pt = old_points[oi]
        new_map_pt = new_points[ni]
        old_px = old_region.from_map((float(old_map_pt.get("lat", 0)), float(old_map_pt.get("lng", 0))))
        new_px = new_region.from_map((float(new_map_pt.get("lat", 0)), float(new_map_pt.get("lng", 0))))
        if old_px is None or new_px is None:
            continue
        seeds_old.append(np.asarray(old_px, dtype=np.float32))
        seeds_new.append(np.asarray(new_px, dtype=np.float32))
    return seeds_old, seeds_new


def _estimate_seed_transform(seeds_old: List[np.ndarray], seeds_new: List[np.ndarray]) -> Tuple[Optional[np.ndarray], int]:
    if len(seeds_old) < 3:
        return None, 0
    old_pts = np.stack(seeds_old, axis=0)
    new_pts = np.stack(seeds_new, axis=0)
    transform, inliers = cv2.estimateAffinePartial2D(
        old_pts,
        new_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=SEED_RANSAC_THRESHOLD,
        confidence=0.999,
    )
    if transform is None or inliers is None:
        return None, 0
    return transform, int(inliers.ravel().astype(bool).sum())


def _filter_by_transform(
    old_pts: np.ndarray,
    new_pts: np.ndarray,
    transform: Optional[np.ndarray],
    threshold: float,
) -> np.ndarray:
    if transform is None:
        return np.ones(len(old_pts), dtype=bool)
    predicted = (old_pts @ transform[:, :2].T) + transform[:, 2]
    residuals = np.linalg.norm(predicted - new_pts, axis=1)
    return residuals < threshold


def _filter_with_ransac(old_pts: np.ndarray, new_pts: np.ndarray, threshold: float) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    if len(old_pts) < 3:
        return None, np.zeros(len(old_pts), dtype=bool), np.full(len(old_pts), float("inf"))
    model, inliers = cv2.estimateAffinePartial2D(
        old_pts,
        new_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.999,
    )
    if model is None or inliers is None:
        return None, np.zeros(len(old_pts), dtype=bool), np.full(len(old_pts), float("inf"))
    mask = inliers.ravel().astype(bool)
    residuals = np.full(len(old_pts), float("inf"), dtype=float)
    if mask.any():
        predicted = (old_pts[mask] @ model[:, :2].T) + model[:, 2]
        residuals[mask] = np.linalg.norm(predicted - new_pts[mask], axis=1)
    return model, mask, residuals


def _run_loftr_matching(old_region: Region, new_region: Region) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    model = _get_loftr_model()
    tensor_old, scale_old = _prepare_loftr_input(old_region)
    tensor_new, scale_new = _prepare_loftr_input(new_region)

    with torch.no_grad():
        output = model({"image0": tensor_old, "image1": tensor_new})

    keypoints0 = output["keypoints0"]
    keypoints1 = output["keypoints1"]
    confidence = output["confidence"]

    if keypoints0.numel() == 0 or keypoints1.numel() == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            {
                "kp_old": 0,
                "kp_new": 0,
                "matches": 0,
                "good": 0,
            },
        )

    kpts_old = (keypoints0.cpu().numpy().astype(np.float32)) / max(scale_old, 1e-6)
    kpts_new = (keypoints1.cpu().numpy().astype(np.float32)) / max(scale_new, 1e-6)
    conf = confidence.cpu().numpy().astype(np.float32)

    # respect polygon masks if present
    if old_region.mask is not None:
        mask_values = old_region.mask.clip(0, 1)
        inside = []
        for pt in kpts_old:
            x, y = int(round(pt[0])), int(round(pt[1]))
            inside.append(0 <= x < mask_values.shape[1] and 0 <= y < mask_values.shape[0] and mask_values[int(y), int(x)] > 0)
        inside = np.asarray(inside, dtype=bool)
        kpts_old, kpts_new, conf = kpts_old[inside], kpts_new[inside], conf[inside]
    if new_region.mask is not None:
        mask_values = new_region.mask.clip(0, 1)
        inside = []
        for pt in kpts_new:
            x, y = int(round(pt[0])), int(round(pt[1]))
            inside.append(0 <= x < mask_values.shape[1] and 0 <= y < mask_values.shape[0] and mask_values[int(y), int(x)] > 0)
        inside = np.asarray(inside, dtype=bool)
        kpts_old, kpts_new, conf = kpts_old[inside], kpts_new[inside], conf[inside]

    return (
        kpts_old,
        kpts_new,
        conf,
        {
            "kp_old": int(keypoints0.shape[0]),
            "kp_new": int(keypoints1.shape[0]),
            "matches": int(keypoints0.shape[0]),
        },
    )


def _apply_affine(pt: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (pt @ transform[:, :2].T) + transform[:, 2]


def _deduplicate_suggestions_list(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in suggestions:
        key = (
            round(item["old"]["lat"], 6),
            round(item["old"]["lng"], 6),
            round(item["new"]["lat"], 6),
            round(item["new"]["lng"], 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _enforce_min_distance(
    suggestions: List[Dict[str, Any]],
    min_distance_m: float,
) -> List[Dict[str, Any]]:
    if min_distance_m <= 0:
        return suggestions
    filtered: List[Dict[str, Any]] = []
    for suggestion in suggestions:
        new_coords = suggestion.get("new")
        if not new_coords or "lat" not in new_coords or "lng" not in new_coords:
            filtered.append(suggestion)
            continue
        keep = True
        for kept in filtered:
            kept_coords = kept.get("new")
            if not kept_coords or "lat" not in kept_coords or "lng" not in kept_coords:
                continue
            dist = _haversine(
                float(new_coords["lat"]),
                float(new_coords["lng"]),
                float(kept_coords["lat"]),
                float(kept_coords["lng"]),
            )
            if dist < min_distance_m:
                keep = False
                break
        if keep:
            filtered.append(suggestion)
    return filtered


def _seed_ncc_expansion(
    old_region: Region,
    new_region: Region,
    seeds_old: List[np.ndarray],
    seed_transform: Optional[np.ndarray],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    diag = {"seed_expanded": 0}
    if seed_transform is None or not seeds_old:
        return [], diag
    if new_region.from_map is None or old_region.to_map is None:
        return [], diag

    old_gray = cv2.cvtColor(old_region.image, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_region.image, cv2.COLOR_BGR2GRAY)

    offsets = [
        (0, 0),
        (90, 0),
        (-90, 0),
        (0, 90),
        (0, -90),
        (70, 70),
        (-70, 70),
        (70, -70),
        (-70, -70),
    ]
    window_radius = 110
    template_size = 72
    half_template = template_size // 2

    suggestions: List[Dict[str, Any]] = []
    for seed_old in seeds_old:
        for dx, dy in offsets:
            center_old = seed_old + np.array([dx, dy], dtype=np.float32)
            x0, y0 = int(round(center_old[0])), int(round(center_old[1]))
            if (
                x0 - half_template < 0
                or y0 - half_template < 0
                or x0 + half_template >= old_gray.shape[1]
                or y0 + half_template >= old_gray.shape[0]
            ):
                continue
            template = old_gray[y0 - half_template : y0 + half_template, x0 - half_template : x0 + half_template]
            if template.size == 0:
                continue
            predicted_new = _apply_affine(center_old.reshape(1, 2), seed_transform)[0]

            search_left = int(round(predicted_new[0])) - window_radius
            search_top = int(round(predicted_new[1])) - window_radius
            search_right = search_left + window_radius * 2 + template.shape[1]
            search_bottom = search_top + window_radius * 2 + template.shape[0]

            search_left = max(0, search_left)
            search_top = max(0, search_top)
            search_right = min(new_gray.shape[1], search_right)
            search_bottom = min(new_gray.shape[0], search_bottom)

            if search_right - search_left < template.shape[1] or search_bottom - search_top < template.shape[0]:
                continue

            search_roi = new_gray[search_top:search_bottom, search_left:search_right]
            result = cv2.matchTemplate(search_roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val < 0.68:
                continue

            match_x = search_left + max_loc[0] + template.shape[1] / 2.0
            match_y = search_top + max_loc[1] + template.shape[0] / 2.0

            old_lat, old_lng = old_region.to_map((float(center_old[0]), float(center_old[1])))
            new_lat, new_lng = new_region.to_map((float(match_x), float(match_y)))

            suggestions.append(
                {
                    "id": f"sugg-seed-{uuid.uuid4().hex[:10]}",
                    "name": f"Seed NCC {len(suggestions) + 1}",
                    "score": float(max_val),
                    "source": "seed-ncc",
                    "old": {"lat": old_lat, "lng": old_lng},
                    "new": {"lat": new_lat, "lng": new_lng},
                    "meta": {"ncc": float(max_val)},
                }
            )
            diag["seed_expanded"] += 1
    return suggestions, diag


def _geocode_name(name: str, bounds: Dict[str, float]) -> Optional[Tuple[float, float]]:
    if not name:
        return None
    params = {
        "q": name,
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
        "bounded": 1,
        "viewbox": f"{bounds['west']},{bounds['north']},{bounds['east']},{bounds['south']}",
    }
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=6,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None
    data = resp.json()
    if not data:
        return None
    try:
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
    except (KeyError, ValueError, IndexError):
        return None
    if not (bounds["south"] <= lat <= bounds["north"] and bounds["west"] <= lon <= bounds["east"]):
        return None
    return lat, lon


def _ocr_geocode_suggestions(
    old_region: Region,
    new_region: Region,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    diag = {"ocr_candidates": 0, "ocr_matches": 0}
    if pytesseract is None:
        return [], diag
    bounds = new_region.info.get("bounds")
    if not bounds:
        return [], diag

    gray = cv2.cvtColor(old_region.image, cv2.COLOR_BGR2GRAY)
    # light threshold to emphasise labels
    gray = cv2.equalizeHist(gray)
    data = pytesseract.image_to_data(gray, output_type=TesseractOutput.DICT, lang="eng+deu")
    suggestions: List[Dict[str, Any]] = []
    seen_names = set()
    for i in range(len(data["text"])):
        conf = int(data["conf"][i]) if data["conf"][i].isdigit() else -1
        text = data["text"][i].strip()
        if conf < 65 or len(text) < 3:
            continue
        text_norm = "".join(ch for ch in text if ch.isalpha())
        if len(text_norm) < 3:
            continue
        if text_norm.lower() in seen_names:
            continue
        seen_names.add(text_norm.lower())
        diag["ocr_candidates"] += 1
        geo = _geocode_name(text_norm, bounds)
        if not geo:
            continue
        lat, lon = geo
        new_px = new_region.from_map((lat, lon))
        if new_px is None:
            continue
        old_px = (
            data["left"][i] + data["width"][i] / 2.0,
            data["top"][i] + data["height"][i] / 2.0,
        )
        old_lat, old_lng = old_region.to_map((float(old_px[0]), float(old_px[1])))
        new_lat, new_lng = new_region.to_map((float(new_px[0]), float(new_px[1])))
        suggestions.append(
            {
                "id": f"sugg-ocr-{uuid.uuid4().hex[:10]}",
                "name": text_norm,
                "score": 0.95,
                "source": "ocr",
                "old": {"lat": old_lat, "lng": old_lng},
                "new": {"lat": new_lat, "lng": new_lng},
                "meta": {"kind": "place-name", "confidence": conf},
            }
        )
        diag["ocr_matches"] += 1
        if diag["ocr_matches"] >= 12:
            break
    return suggestions, diag


def _format_suggestions(
    old_region: Region,
    new_region: Region,
    old_pts: np.ndarray,
    new_pts: np.ndarray,
    confidences: np.ndarray,
    residuals: np.ndarray,
) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    if old_pts.size == 0:
        return suggestions
    order = np.argsort(-confidences)
    for idx in order[:SUGGESTION_LIMIT]:
        old_px = old_pts[idx]
        new_px = new_pts[idx]
        score = float(confidences[idx])
        residual = float(residuals[idx]) if np.isfinite(residuals[idx]) else None
        old_lat, old_lng = old_region.to_map((float(old_px[0]), float(old_px[1])))
        new_lat, new_lng = new_region.to_map((float(new_px[0]), float(new_px[1])))
        suggestions.append(
            {
                "id": f"sugg-{uuid.uuid4().hex[:10]}",
                "name": f"Match {len(suggestions) + 1}",
                "score": score,
                "source": "loftr",
                "old": {"lat": old_lat, "lng": old_lng},
                "new": {"lat": new_lat, "lng": new_lng},
                "meta": {"residual": residual},
            }
        )
    return suggestions


def generate_suggestions(payload: Dict[str, Any]) -> Dict[str, Any]:
    if cv2 is None:
        raise HTTPException(status_code=500, detail="OpenCV is required to generate suggestions.")
    if torch is None:
        raise HTTPException(status_code=500, detail="PyTorch is required to generate suggestions.")

    started = time.time()
    state = payload.get("state") or {}
    polygons = state.get("polygons") or {}
    map_sources = payload.get("map_sources") or {}
    view = payload.get("view") or {}
    constraints = payload.get("constraints") or {}
    req_min_distance = max(0.0, float(constraints.get("min_distance") or DEFAULT_MIN_DISTANCE_METERS))
    req_max_suggestions = int(constraints.get("max_suggestions") or SUGGESTION_LIMIT)
    req_max_suggestions = max(1, min(SUGGESTION_LIMIT, req_max_suggestions))

    old_region = _load_upload_region(map_sources.get("old") or {}, polygons.get("old") or [], "old")
    if (map_sources.get("new") or {}).get("type", "osm").lower() == "osm":
        new_region = _load_osm_region(map_sources.get("new") or {}, view.get("new") or {}, polygons.get("new") or [])
    else:
        new_region = _load_upload_region(map_sources.get("new") or {}, polygons.get("new") or [], "new")

    seeds_old, seeds_new = _extract_seed_pairs(payload, old_region, new_region)
    seed_transform, seed_inliers = _estimate_seed_transform(seeds_old, seeds_new)

    kpts_old, kpts_new, confidences, diag = _run_loftr_matching(old_region, new_region)
    diag.update({"good": 0, "min_distance": req_min_distance, "max_suggestions": req_max_suggestions})  # will be filled below

    if kpts_old.size == 0 or kpts_new.size == 0:
        elapsed = int((time.time() - started) * 1000)
        diag.update(
            {
                "good": 0,
                "inliers": 0,
                "seed_count": len(seeds_old),
                "seed_inliers": seed_inliers,
                "seed_expanded": 0,
                "ocr_candidates": 0,
                "ocr_matches": 0,
                "min_distance": req_min_distance,
                "max_suggestions": req_max_suggestions,
                "elapsed_ms": elapsed,
            }
        )
        return {
            "request_id": str(uuid.uuid4()),
            "count": 0,
            "mode": payload.get("mode", "bulk"),
            "message": "No suggestions found for the current selection.",
            "suggestions": [],
            "diagnostics": diag,
        }

    conf_mask = confidences >= CONFIDENCE_THRESHOLD
    kpts_old = kpts_old[conf_mask]
    kpts_new = kpts_new[conf_mask]
    confidences = confidences[conf_mask]
    diag["good"] = int(len(confidences))

    if seed_transform is not None and kpts_old.size > 0:
        consistent_mask = _filter_by_transform(kpts_old, kpts_new, seed_transform, SEED_CONSISTENCY_PX)
        kpts_old = kpts_old[consistent_mask]
        kpts_new = kpts_new[consistent_mask]
        confidences = confidences[consistent_mask]
        diag["seed_consistent"] = int(consistent_mask.sum())
    else:
        diag["seed_consistent"] = int(len(confidences))

    ransac_model, inlier_mask, residuals = _filter_with_ransac(kpts_old, kpts_new, RANSAC_THRESHOLD)
    diag["inliers"] = int(inlier_mask.sum())
    diag["ransac_fallback"] = 0

    if inlier_mask.any():
        filtered_old = kpts_old[inlier_mask]
        filtered_new = kpts_new[inlier_mask]
        filtered_conf = confidences[inlier_mask]
        filtered_residuals = residuals[inlier_mask]
        suggestions_loftr = _format_suggestions(
            old_region,
            new_region,
            filtered_old,
            filtered_new,
            filtered_conf,
            filtered_residuals,
        )
    else:
        top_count = min(RANSAC_FALLBACK_TOPK, len(confidences))
        if top_count > 0:
            top_idx = np.argsort(confidences)[::-1][:top_count]
            fallback_old = kpts_old[top_idx]
            fallback_new = kpts_new[top_idx]
            fallback_conf = confidences[top_idx]
            fallback_residuals = np.full(top_count, float("inf"), dtype=float)
            suggestions_loftr = _format_suggestions(
                old_region,
                new_region,
                fallback_old,
                fallback_new,
                fallback_conf,
                fallback_residuals,
            )
            diag["ransac_fallback"] = int(top_count)
        else:
            suggestions_loftr = []

    seed_suggestions, seed_diag = _seed_ncc_expansion(old_region, new_region, seeds_old, seed_transform)
    ocr_suggestions, ocr_diag = _ocr_geocode_suggestions(old_region, new_region)

    combined = suggestions_loftr + seed_suggestions + ocr_suggestions
    combined = _deduplicate_suggestions_list(combined)
    combined = _enforce_min_distance(combined, req_min_distance)
    if len(combined) > req_max_suggestions:
        combined = combined[:req_max_suggestions]

    elapsed = int((time.time() - started) * 1000)
    diag.update(
        {
            "seed_count": len(seeds_old),
            "seed_inliers": seed_inliers,
            "seed_expanded": seed_diag.get("seed_expanded", 0),
            "ocr_candidates": ocr_diag.get("ocr_candidates", 0),
            "ocr_matches": ocr_diag.get("ocr_matches", 0),
            "min_distance": req_min_distance,
            "max_suggestions": req_max_suggestions,
            "elapsed_ms": elapsed,
        }
    )

    if combined:
        message = f"{len(combined)} suggestion{'s' if len(combined) != 1 else ''} ready."
    else:
        message = "No suggestions found for the current selection."

    return {
        "request_id": str(uuid.uuid4()),
        "mode": payload.get("mode", "bulk"),
        "count": len(combined),
        "message": message,
        "suggestions": combined,
        "diagnostics": diag,
    }
DEFAULT_MIN_DISTANCE_METERS = 0.0
