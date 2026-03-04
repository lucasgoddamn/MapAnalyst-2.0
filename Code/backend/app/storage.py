from __future__ import annotations

from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OLD_UPLOAD_DIR = UPLOAD_DIR / "old"
NEW_UPLOAD_DIR = UPLOAD_DIR / "new"
OSM_CACHE_DIR = DATA_DIR / "osm_cache"

for path in (OLD_UPLOAD_DIR, NEW_UPLOAD_DIR, OSM_CACHE_DIR):
  path.mkdir(parents=True, exist_ok=True)


def resolve_storage_path(relative_path: str) -> Path:
  """
  Resolve a relative path (as returned to the client) to an absolute filesystem path.
  Raises ValueError if the path escapes the data directory.
  """
  if not relative_path:
    raise ValueError("Storage path is empty.")
  candidate = (DATA_DIR / relative_path).resolve()
  if DATA_DIR not in candidate.parents and candidate != DATA_DIR:
    raise ValueError("Invalid storage path outside of data directory.")
  return candidate
