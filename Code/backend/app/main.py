from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path
import time
import uuid
from .compute.transforms import solve_helmert4, solve_affine6, solve_affine5_like
from .compute.robust import solve_robust_helmert
from .compute.grid import build_distortion_grid
from .compute.isolines import build_isolines
from .suggestions import generate_suggestions
from .storage import DATA_DIR, OLD_UPLOAD_DIR, NEW_UPLOAD_DIR
import math


class XY(BaseModel):
    x: float
    y: float


class Pair(BaseModel):
    name: str = ""
    old: XY
    new: XY


class RobustParams(BaseModel):
    huber: Dict[str, float] = Field(default_factory=lambda: {"k": 1.5})
    v: Dict[str, float] = Field(default_factory=lambda: {"k": 1.5, "e": 0.6})
    hampel: Dict[str, float] = Field(default_factory=lambda: {"a": 1.0, "b": 2.0, "c": 4.0})


class ComputeRequest(BaseModel):
    mode: Literal["helmert4", "affine5", "affine6", "robust"]
    robustKind: Literal["huber", "v", "hampel"] = "huber"
    params: RobustParams = RobustParams()
    target: Literal["old", "new"] = "old"
    pairs: List[Pair]
    settings: Optional[dict] = None


class ParamOut(BaseModel):
    name: str
    value: float


class ResidualOut(BaseModel):
    name: str
    old: XY
    new: XY
    vx: float
    vy: float
    vnorm: float


class ComputeResponse(BaseModel):
    mode: str
    target: str
    n_pairs: int
    parameters: List[ParamOut]
    rmse: float
    residuals: List[ResidualOut]
    meta: dict = {}


app = FastAPI(title="MapAnalyst Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _storage_response(original_name: str, stored_path: Path) -> Dict[str, Any]:
    return {
        "filename": original_name,
        "stored_as": str(stored_path.relative_to(DATA_DIR)).replace("\\", "/"),
    }


async def _persist_upload(file: UploadFile, category: Literal["old", "new"]) -> Dict[str, Any]:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    ext = Path(file.filename or "").suffix or ".bin"
    filename = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}{ext}"
    target_dir = OLD_UPLOAD_DIR if category == "old" else NEW_UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    target_path.write_bytes(contents)
    file.file.seek(0)
    return _storage_response(file.filename or filename, target_path)


def _prep_xy(req: ComputeRequest):
    src_x, src_y, trg_x, trg_y, names = [], [], [], [], []
    for p in req.pairs:
        if req.target == "old":
            sx, sy = p.new.x, p.new.y
            tx, ty = p.old.x, p.old.y
        else:
            sx, sy = p.old.x, p.old.y
            tx, ty = p.new.x, p.new.y
        src_x.append(sx)
        src_y.append(sy)
        trg_x.append(tx)
        trg_y.append(ty)
        names.append(p.name or "")
    return src_x, src_y, trg_x, trg_y, names


@app.post("/api/compute", response_model=ComputeResponse)
def compute(req: ComputeRequest):
    src_x, src_y, trg_x, trg_y, names = _prep_xy(req)
    n = len(src_x)
    if n == 0:
        return ComputeResponse(
            mode=req.mode,
            target=req.target,
            n_pairs=0,
            parameters=[],
            rmse=0.0,
            residuals=[],
            meta={"message": "no pairs"},
        )

    if req.mode == "helmert4":
        sol = solve_helmert4(src_x, src_y, trg_x, trg_y)
    elif req.mode == "affine6":
        sol = solve_affine6(src_x, src_y, trg_x, trg_y)
    elif req.mode == "affine5":
        sol = solve_affine5_like(src_x, src_y, trg_x, trg_y)
    elif req.mode == "robust":
        sol = solve_robust_helmert(src_x, src_y, trg_x, trg_y, kind=req.robustKind, params=req.params.dict())
    else:
        raise ValueError("Unsupported mode")

    params_out = [ParamOut(name=k, value=float(v)) for k, v in sol["params"].items()]
    resids_out: List[ResidualOut] = []
    rmse = 0.0
    for i in range(n):
        vx = float(sol["residuals_x"][i])
        vy = float(sol["residuals_y"][i])
        vnorm = math.hypot(vx, vy)
        rmse += vnorm * vnorm
        resids_out.append(
            ResidualOut(
                name=names[i],
                old=req.pairs[i].old,
                new=req.pairs[i].new,
                vx=vx,
                vy=vy,
                vnorm=vnorm,
            )
        )
    rmse = math.sqrt(rmse / max(n, 1))

    return ComputeResponse(
        mode=req.mode,
        target=req.target,
        n_pairs=n,
        parameters=params_out,
        rmse=rmse,
        residuals=resids_out,
        meta=sol.get("meta", {}),
    )


@app.post("/api/visual/distortion-grid")
async def visual_distortion_grid(payload: Dict[str, Any]):
    try:
        return build_distortion_grid(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Distortion grid failed: {exc}")


@app.post("/api/visual/isolines")
async def visual_isolines(payload: Dict[str, Any]):
    try:
        return build_isolines(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Isolines failed: {exc}")


@app.post("/api/suggest_links")
def suggest_links(payload: Dict[str, Any]):
    try:
        return generate_suggestions(payload)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Suggestion pipeline failed: {exc}") from exc


@app.post("/api/upload_old_map")
async def upload_old_map(file: UploadFile = File(...)):
    try:
        return await _persist_upload(file, "old")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store old map: {exc}") from exc


@app.post("/api/upload_new_map")
async def upload_new_map(file: UploadFile = File(...)):
    try:
        return await _persist_upload(file, "new")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store new map: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)