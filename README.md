# MapAnalyst 2.0

This repository contains the implementation and supporting material for the master thesis project "MapAnalyst 2.0", a web-based continuation of MapAnalyst with AI-assisted control point suggestions for comparing historical and modern maps.

The repository is structured into:

- `Code/`: the web application and backend service
- `Test/`: sample maps and test datasets

## What the application does

MapAnalyst 2.0 brings the core MapAnalyst workflow into the browser:

- load and compare an old map with a modern reference map
- create and manage linked control points
- compute transformation models (Helmert, affine, and robust variants)
- generate visual analysis outputs such as distortion grids and isolines
- suggest possible linked control points with a computer-vision-based backend

The frontend is built with Astro, React, and Tailwind CSS. The backend is a FastAPI service with image-processing and ML-related dependencies such as OpenCV, PyTorch, and Kornia.

## Repository structure

```text
masterarbeit/
|- Code/
|  |- src/                  # Astro frontend source
|  |- public/               # static assets
|  |- backend/              # FastAPI backend
|  |- package.json          # frontend scripts and dependencies
|  `- README.md             # application-specific setup notes
`- Test/                    # sample map pairs and linked point examples
```

## Local development

The application runs as two local services:

- frontend: Astro dev server on `http://localhost:4321`
- backend: FastAPI on `http://localhost:8000`

The frontend currently calls the backend directly at `http://localhost:8000`, so both services must be running at the same time for the analysis features to work.

### Prerequisites

- Current Node.js LTS version
- Python 3.9+ recommended
- npm (bundled with Node.js)

### 1. Start the frontend

Open a terminal in the repository root and run:

```powershell
cd Code
npm install
npm run dev
```

Astro will start the local development server. By default, it is available at `http://localhost:4321`.

### 2. Start the backend

Open a second terminal and run:

```powershell
cd Code/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If you are not using PowerShell:

- Windows Command Prompt: `.\.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

The backend will then be available at `http://localhost:8000`.

Alternative: because `app/main.py` starts Uvicorn in its `__main__` block, you can also launch the backend like this from `Code/backend`:

```powershell
python -m app.main
```

That starts the same FastAPI app on `127.0.0.1:8000` with reload enabled.

Important: `python -m app.main` uses whichever Python interpreter the `python` command currently points to. If you want to use the repository's existing `venv`, either activate it first or call it directly:

```powershell
cd Code/backend
.\venv\Scripts\python.exe -m app.main
```

If you prefer a Windows batch launcher, use a path-relative version instead of a hardcoded absolute path. Assuming the `.bat` file is stored in the repository root, this is a safe version:

```bat
@echo off
start "" cmd /k "cd /d ""%~dp0Code"" && npm run dev"
start "" cmd /k "cd /d ""%~dp0Code\backend"" && ""%~dp0Code\backend\venv\Scripts\python.exe"" -m app.main"
```

`%~dp0` expands to the folder where the batch file itself is located, so the script still works if the repository is moved to another drive or directory.

### 3. Open the application

Once both processes are running, open:

`http://localhost:4321/mapanalyst`

You can also start from `http://localhost:4321` and navigate there through the UI. The main analysis workflow is available on the MapAnalyst page.

## Build a production frontend bundle

To create a production build of the frontend:

```powershell
cd Code
npm run build
```

To preview the built frontend locally:

```powershell
cd Code
npm run preview
```

## Backend notes

- Uploaded map files are stored under `Code/backend/app/data/uploads/`.
- Cached backend data is stored under `Code/backend/app/data/`.
- Installing backend dependencies can take a while because `torch` and `torchvision` are included.
- The current backend exposes endpoints for compute, visualization, suggestions, and map uploads under `/api/...`.

## Sample data

Example data for testing the workflow is available in:

- `Test/Maps/Basel/`
- `Test/Maps/Warsaw/`

These folders contain example raster maps and linked point files that are useful for manual testing and demonstrations.

## Project status

This repository reflects a master thesis prototype. The app is functional as a local research/demo environment, and parts of the codebase still include starter-template remnants.

## License

The `Code/` subproject includes `Code/LICENSE` (MIT), inherited from the original frontend starter template. There is currently no separate top-level license file for the full repository contents.
