Preprocess Package

Layout
- src/preprocess/: Python package with all preprocessing code
- cleaned/, events/, heatmaps/, scanpaths/, summaries/: generated outputs
- params.yaml: located beside the package in src/preprocess/ (loaded by default)

Usage
- Windows PowerShell:
  - `$env:PYTHONPATH='code\preprocess\src'; python -m preprocess.pipeline --input-csv path\to\samples.csv`
- Linux/WSL:
  - `PYTHONPATH=code/preprocess/src python3 -m preprocess.pipeline --input-csv path/to/samples.csv`

Outputs
- Written under this folder (code/preprocess/...) to keep code and artifacts together.

