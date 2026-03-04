# MapAnalyst 2.0 Application (`Code/`)

This folder contains the runnable web application of the thesis project:

- Astro frontend in `src/`
- FastAPI backend in `backend/`

## Prerequisites

- Node.js (current LTS)
- Python 3.9+

## Run locally

You need two terminals.

### 1) Frontend

```powershell
cd Code
npm install
npm run dev
```

Frontend runs on `http://localhost:4321`.

### 2) Backend

```powershell
cd Code/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend runs on `http://localhost:8000`.

The frontend calls backend endpoints on `http://localhost:8000`, so both services must run together.

## Alternative backend start

`app/main.py` contains a `__main__` launcher, so this also works:

```powershell
cd Code/backend
.\venv\Scripts\python.exe -m app.main
```

## Build frontend

```powershell
cd Code
npm run build
npm run preview
```

## Notes

- Uploaded files and backend cache are under `backend/app/data/`.
- This repository root has additional thesis assets (`TextMa/`, `Test/`, `VortragMa/`). See the root README for full project context.
