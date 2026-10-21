"""Where the data and the results live (absolute paths, so scripts run from any directory)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "li_ma_2020"
RESULTS_DIR = REPO_ROOT / "results"
FITS_DIR = RESULTS_DIR / "fits"                 # new fit runs, one folder each
PAPER_FITS_DIR = RESULTS_DIR / "paper_fits"     # the fits reported in the paper, one folder per model
FIGURES_DIR = REPO_ROOT / "figures"             # figures/paper from the paper's fits; figures/runs, figures/smoke from new runs
