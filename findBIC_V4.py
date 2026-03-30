import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import datetime as dt
import importlib.util
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
try:
    import scienceplots  # noqa: F401
    HAS_SCIENCEPLOTS = True
except ModuleNotFoundError:
    HAS_SCIENCEPLOTS = False
try:
    from skopt import Optimizer
    from skopt.space import Categorical, Real
    SKOPT_IMPORT_ERROR = ""
except ModuleNotFoundError as exc:
    Optimizer = None
    Categorical = None
    Real = None
    SKOPT_IMPORT_ERROR = str(exc)

import legume


# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------

# Default polarization family. 1 -> TM, 2 -> TE.
CHOOSE_MODE = 1  # Default to TM-like family; set to 2 for TE.

# Target wavelength window and material indices.
DEFAULT_RANGE1_NM = 520.0  # Lower edge of the target wavelength window in nm.
DEFAULT_RANGE2_NM = 540.0  # Upper edge of the target wavelength window in nm.
DEFAULT_N_SLAB = 2.0  # Refractive index of the slab material.
DEFAULT_N_HOLE = 1.49  # Refractive index inside the holes/background inclusion.

# Scan modes:
# a_r_d -> scan a, hole radius ratio r/a, and thickness ratio d/a
# a_r   -> fix thickness ratio d/a, scan a and hole radius ratio r/a
# a_d   -> fix hole radius ratio r/a, scan a and thickness ratio d/a
# r_d   -> fix a, scan hole radius ratio r/a and thickness ratio d/a
DEFAULT_SCAN_MODE = "a_d"  # Scan a and thickness ratio d/a while keeping hole radius ratio r/a fixed.

# Fixed values used by the 2D scan modes.
DEFAULT_FIXED_A_NM = 300.0  # Fixed period a for r_d mode.
DEFAULT_FIXED_R_OVER_A = 0.32  # Fixed hole radius ratio r/a for a_d mode.
DEFAULT_FIXED_D_OVER_A = 1.0 / 3.0  # Fixed thickness ratio d/a for a_r mode.

# Scan ranges and sampling density.
DEFAULT_R_MIN_OVER_A = 0.30  # Minimum r/a when radius ratio is scanned.
DEFAULT_R_MAX_OVER_A = 0.36  # Maximum r/a when radius ratio is scanned.
DEFAULT_R_POINTS = 4  # Number of r/a samples between r_min_over_a and r_max_over_a.
DEFAULT_D_MIN = 0.1  # Minimum d/a when thickness ratio is scanned.
DEFAULT_D_MAX = 2.5  # Maximum d/a when thickness ratio is scanned.
DEFAULT_D_POINTS = 30  # Number of d/a samples between d_min and d_max.
DEFAULT_MIN_H_NM = 60.0  # Reject geometries with slab thickness h below this value.
DEFAULT_A_POINTS = 20  # Number of a samples per adaptive a-window.
DEFAULT_A_EXPAND_ROUNDS = 6  # Number of adaptive a-window expansion rounds.

# Bayesian optimization settings.
DEFAULT_BO_CALLS = 45  # Number of Bayesian-optimization evaluations after coarse scan.
DEFAULT_BO_RANDOM_STARTS = 10  # Random warm-up evaluations for Bayesian optimization.
DEFAULT_BO_ROUNDS = 3  # Number of BO batches used for candidate screening.
DEFAULT_COARSE_BOOTSTRAP_POINTS = 36  # Maximum coarse bootstrap points per a-window before BO takes over.
DEFAULT_REFINE_BO_CALLS_PER_CENTER = 4  # Local BO evaluations per shortlisted center during refinement.
DEFAULT_REFINE_BO_RANDOM_STARTS = 2  # Random warm-up evaluations inside each local-refinement BO run.

# Parallel runtime settings.
DEFAULT_WORKERS = 0  # 0 means auto; 1 disables multiprocessing.
DEFAULT_THREADS_PER_WORKER = 1  # Keep BLAS/OpenMP threads low inside each worker.
DEFAULT_PARALLEL_CHUNK_SIZE = 0  # 0 means auto chunk size.

# Plot title spacing.
DEFAULT_AX_TITLE_PAD = 14.0  # Keep subplot titles a bit farther from axes for clean screenshots.
DEFAULT_SUPTITLE_Y = 1.03  # Lift the figure title above the plotting area.

# Numerical solver settings.
DEFAULT_GMAX = 4.2  # Main plane-wave cutoff used during screening.
DEFAULT_NUMEIG_GAMMA = 14  # Number of Gamma-point eigenmodes to compute in screening.
DEFAULT_PATH_KMAX = 0.07  # Max reduced wavevector for the final validation path.
DEFAULT_PATH_POINTS_PER_SIDE = 31  # Samples on each side of Gamma for the validation path.
DEFAULT_NEAR_GAMMA_POINTS_PER_SIDE = 5  # Samples on each side of Gamma for near-Gamma scoring.
DEFAULT_VALIDATE_GMAX = 5.5  # Larger cutoff used during final validation.
DEFAULT_NUMEIG_PATH = 10  # Number of bands retained along the k-path.

# Refinement and candidate-selection settings.
DEFAULT_REFINE_FACTOR = 2  # Local densification factor around shortlisted candidates.
DEFAULT_MAX_REFINE_CENTERS = 6  # Maximum number of coarse candidates sent to refinement.
DEFAULT_MAX_NEAR_GAMMA_EVALS = 12  # Upper bound on expensive near-Gamma reevaluations.

# Output defaults.
DEFAULT_PROJECT_NAME = "findSi3N4BIC_V4"  # Prefix used for output folders and report files.


# -----------------------------------------------------------------------------
# Basic config helpers
# -----------------------------------------------------------------------------


def choose_mode_to_pol(choose_mode: int) -> str:
    if choose_mode == 1:
        return "TM"
    if choose_mode == 2:
        return "TE"
    raise ValueError("CHOOSE_MODE must be 1 (TM) or 2 (TE)")


# User-tunable screening and plotting thresholds.
DEFAULT_Q_THRESHOLD = 1e6
DEFAULT_IM_THRESHOLD = 1e-6
DEFAULT_DISPLAY_MARGIN_NM = 20.0
DEFAULT_PURCELL_WEIGHT = 2.5  # Small tie-break weight for the approximate Purcell metric.
DEFAULT_PURCELL_TOP_EVALS = 6  # Max candidate count per stage for the Purcell estimate.
DEFAULT_PURCELL_XY_SAMPLES = 61  # In-plane field grid used by the slab-confined Purcell estimate.
DEFAULT_PURCELL_Z_SAMPLES = 13  # Vertical field slices used by the slab-confined Purcell estimate.
DEFAULT_PURCELL_LOG_CAP = 8.0  # Keep extreme Purcell values from dominating ranking scores.
DEFAULT_ROBUST_TOP_CANDIDATES = 3  # Final shortlist size used for fabrication-tolerance screening.
DEFAULT_ROBUST_SAMPLES = 12  # Monte Carlo fabrication samples per final candidate.
DEFAULT_ROBUST_A_SIGMA_NM = 2.0  # 1-sigma period error used in the robustness screen.
DEFAULT_ROBUST_RADIUS_SIGMA_NM = 2.0  # 1-sigma hole-radius error used in the robustness screen.
DEFAULT_ROBUST_H_SIGMA_NM = 3.0  # 1-sigma slab-thickness error used in the robustness screen.
DEFAULT_ROBUST_BETA = 0.75  # Risk aversion in robust score = mean - beta * std.
DEFAULT_ROBUST_SEED = 0  # Reproducible RNG seed for the fabrication-tolerance screen.


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyConfig:
    # Metadata for one polarization family used by the scan pipeline.
    key: str
    title: str
    pol_label: str
    gmode_inds: Tuple[int, ...]
    color: str


@dataclass(frozen=True)
class ScanPoint:
    # One discrete point in the scan grid.
    a_nm: float
    r_over_a: float
    h_nm: float
    d_over_a: float


@dataclass(frozen=True)
class ScanPointResult:
    # Aggregated Gamma-point metrics for one scan point.
    family_key: str
    a_nm: float
    r_over_a: float
    h_nm: float
    d_over_a: float
    window_mode_count: int
    bic_count: int
    best_band_index: int
    best_freq: float
    best_freq_im: float
    best_q: float
    best_lambda_nm: float
    near_gamma_score: float
    purcell_factor_est: float = math.nan
    mode_volume_a3: float = math.nan
    mode_volume_m3: float = math.nan


@dataclass(frozen=True)
class RobustCandidateSummary:
    # Gamma-only fabrication-tolerance statistics for one final candidate.
    rank: int
    a_nm: float
    r_over_a: float
    h_nm: float
    d_over_a: float
    nominal_score: float
    mean_score: float
    std_score: float
    robust_score: float
    score_q10: float
    yield_fraction: float
    valid_fraction: float
    passing_samples: int
    valid_samples: int
    total_samples: int


FAMILIES: Tuple[FamilyConfig, ...] = (
    FamilyConfig(
        key="mode1_TM",
        title="mode1 TM-like",
        pol_label="TM",
        gmode_inds=tuple(range(1, 12, 2)),
        color="#c62828",
    ),
    FamilyConfig(
        key="mode2_TE",
        title="mode2 TE-like",
        pol_label="TE",
        gmode_inds=tuple(range(0, 12, 2)),
        color="#1565c0",
    ),
)


# -----------------------------------------------------------------------------
# CLI setup
# -----------------------------------------------------------------------------


def get_active_family(pol: str) -> FamilyConfig:
    pol_upper = pol.upper()
    for family in FAMILIES:
        if family.pol_label.upper() == pol_upper:
            return family
    raise ValueError(f"Unsupported polarization '{pol}'. Use TE or TM.")


def parse_args() -> argparse.Namespace:
    # Command-line interface for scan configuration and output control.
    parser = argparse.ArgumentParser(
        description=(
            "Fast BIC search for a square-lattice photonic-crystal slab with "
            "varying a, hole radius ratio r/a, and thickness ratio d/a. The code first does a physically constrained "
            "coarse scan near Gamma, then optionally refines the promising "
            "region and verifies the selected candidate along a short "
            "Gamma-centered path."
        )
    )
    parser.add_argument("--range1-nm", type=float, default=DEFAULT_RANGE1_NM)
    parser.add_argument("--range2-nm", type=float, default=DEFAULT_RANGE2_NM)
    parser.add_argument(
        "--scan-mode",
        type=str,
        choices=["a_r_d", "a_r", "a_d", "r_d"],
        default=DEFAULT_SCAN_MODE,
        help="Choose which variables are scanned: a_r_d, a_r, a_d, or r_d.",
    )
    parser.add_argument(
        "--pol",
        type=str,
        choices=["TE", "TM", "te", "tm"],
        default=choose_mode_to_pol(CHOOSE_MODE),
        help="Only compute one polarization family per run.",
    )
    parser.add_argument("--n-slab", type=float, default=DEFAULT_N_SLAB)
    parser.add_argument("--n-hole", type=float, default=DEFAULT_N_HOLE)
    parser.add_argument("--r-min-over-a", type=float, default=DEFAULT_R_MIN_OVER_A)
    parser.add_argument("--r-max-over-a", type=float, default=DEFAULT_R_MAX_OVER_A)
    parser.add_argument("--r-points", type=int, default=DEFAULT_R_POINTS)
    parser.add_argument(
        "--r-values-over-a",
        type=str,
        default=None,
        help="Optional legacy override, for example 0.30,0.32,0.34,0.36",
    )
    parser.add_argument("--d-min", type=float, default=DEFAULT_D_MIN)
    parser.add_argument("--d-max", type=float, default=DEFAULT_D_MAX)
    parser.add_argument("--d-points", "--h-points", dest="d_points", type=int, default=DEFAULT_D_POINTS)
    parser.add_argument(
        "--fixed-a-nm",
        type=float,
        default=DEFAULT_FIXED_A_NM,
        help="Fixed period a, used only when scan-mode is r_d.",
    )
    parser.add_argument(
        "--fixed-d-over-a",
        type=float,
        default=DEFAULT_FIXED_D_OVER_A,
        help="Fixed thickness ratio d/a, used only when scan-mode is a_r.",
    )
    parser.add_argument(
        "--fixed-r-over-a",
        type=float,
        default=DEFAULT_FIXED_R_OVER_A,
        help="Fixed hole radius ratio r/a, used only when scan-mode is a_d.",
    )
    parser.add_argument("--a-min-nm", type=float, default=None)
    parser.add_argument("--a-max-nm", type=float, default=None)
    parser.add_argument("--a-points", type=int, default=DEFAULT_A_POINTS)
    parser.add_argument("--a-expand-rounds", type=int, default=DEFAULT_A_EXPAND_ROUNDS)
    parser.add_argument(
        "--coarse-bootstrap-points",
        type=int,
        default=DEFAULT_COARSE_BOOTSTRAP_POINTS,
        help="Maximum coarse bootstrap points per a-window before BO takes over.",
    )
    parser.add_argument("--bo-calls", type=int, default=DEFAULT_BO_CALLS)
    parser.add_argument("--bo-random-starts", type=int, default=DEFAULT_BO_RANDOM_STARTS)
    parser.add_argument(
        "--bo-rounds",
        type=int,
        default=DEFAULT_BO_ROUNDS,
        help="Number of BO screening rounds. Previously screened points are skipped in later rounds.",
    )
    parser.add_argument(
        "--refine-bo-calls-per-center",
        type=int,
        default=DEFAULT_REFINE_BO_CALLS_PER_CENTER,
        help="Local BO evaluations per shortlisted center during refinement. Set 0 to fall back to dense local grids.",
    )
    parser.add_argument(
        "--refine-bo-random-starts",
        type=int,
        default=DEFAULT_REFINE_BO_RANDOM_STARTS,
        help="Random warm-up evaluations inside each local-refinement BO run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Process count for heavy scan stages. 0 means auto, 1 means serial.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=DEFAULT_THREADS_PER_WORKER,
        help="BLAS/OpenMP thread count inside each worker process.",
    )
    parser.add_argument(
        "--parallel-chunk-size",
        type=int,
        default=DEFAULT_PARALLEL_CHUNK_SIZE,
        help="Points per submitted worker chunk. 0 means auto.",
    )
    parser.add_argument("--min-h-nm", type=float, default=DEFAULT_MIN_H_NM)
    parser.add_argument(
        "--allow-superwavelength",
        action="store_true",
        help="If set, do not enforce a < range1_nm during period selection.",
    )
    parser.add_argument("--gmax", type=float, default=DEFAULT_GMAX)
    parser.add_argument(
        "--validate-gmax",
        type=float,
        default=DEFAULT_VALIDATE_GMAX,
        help="Higher gmax used only for the near-Gamma dispersion verification plot.",
    )
    parser.add_argument("--numeig-gamma", type=int, default=DEFAULT_NUMEIG_GAMMA)
    parser.add_argument(
        "--numeig-path",
        type=int,
        default=DEFAULT_NUMEIG_PATH,
        help="Number of eigenvalues kept for the near-Gamma verification path.",
    )
    parser.add_argument(
        "--refine-factor",
        type=int,
        default=DEFAULT_REFINE_FACTOR,
        help="How many fine points to insert inside one coarse thickness-ratio step.",
    )
    parser.add_argument(
        "--max-refine-centers",
        type=int,
        default=DEFAULT_MAX_REFINE_CENTERS,
        help="Maximum number of local neighborhoods selected for fine refinement.",
    )
    parser.add_argument(
        "--max-near-gamma-evals",
        type=int,
        default=DEFAULT_MAX_NEAR_GAMMA_EVALS,
        help="At most this many candidates are scored with expensive near-Gamma loss/Q evaluation per stage.",
    )
    parser.add_argument("--q-threshold", type=float, default=DEFAULT_Q_THRESHOLD)
    parser.add_argument("--im-threshold", type=float, default=DEFAULT_IM_THRESHOLD)
    parser.add_argument(
        "--purcell-weight",
        type=float,
        default=DEFAULT_PURCELL_WEIGHT,
        help="Tie-break weight for the slab-confined unit-cell Purcell estimate during candidate ranking.",
    )
    parser.add_argument(
        "--purcell-top-evals",
        type=int,
        default=DEFAULT_PURCELL_TOP_EVALS,
        help="At most this many candidates per stage get the Purcell estimate.",
    )
    parser.add_argument(
        "--purcell-xy-samples",
        type=int,
        default=DEFAULT_PURCELL_XY_SAMPLES,
        help="In-plane field grid used by the Purcell estimate.",
    )
    parser.add_argument(
        "--purcell-z-samples",
        type=int,
        default=DEFAULT_PURCELL_Z_SAMPLES,
        help="Vertical slice count used by the Purcell estimate.",
    )
    parser.add_argument(
        "--robust-top-candidates",
        type=int,
        default=DEFAULT_ROBUST_TOP_CANDIDATES,
        help="How many final candidates receive a fabrication-tolerance robustness screen. Set 0 to disable.",
    )
    parser.add_argument(
        "--robust-samples",
        type=int,
        default=DEFAULT_ROBUST_SAMPLES,
        help="Monte Carlo fabrication samples per shortlisted final candidate.",
    )
    parser.add_argument(
        "--robust-a-sigma-nm",
        type=float,
        default=DEFAULT_ROBUST_A_SIGMA_NM,
        help="1-sigma period fabrication error in nm used by the robustness screen.",
    )
    parser.add_argument(
        "--robust-radius-sigma-nm",
        type=float,
        default=DEFAULT_ROBUST_RADIUS_SIGMA_NM,
        help="1-sigma hole-radius fabrication error in nm used by the robustness screen.",
    )
    parser.add_argument(
        "--robust-h-sigma-nm",
        type=float,
        default=DEFAULT_ROBUST_H_SIGMA_NM,
        help="1-sigma slab-thickness fabrication error in nm used by the robustness screen.",
    )
    parser.add_argument(
        "--robust-beta",
        type=float,
        default=DEFAULT_ROBUST_BETA,
        help="Risk aversion in the robust score: mean_score - beta * std_score.",
    )
    parser.add_argument(
        "--robust-seed",
        type=int,
        default=DEFAULT_ROBUST_SEED,
        help="Random seed used by the fabrication-tolerance robustness screen.",
    )
    parser.add_argument(
        "--display-margin-nm",
        type=float,
        default=DEFAULT_DISPLAY_MARGIN_NM,
        help="Extra wavelength margin around the target window for the dispersion plot.",
    )
    parser.add_argument("--path-kmax", type=float, default=DEFAULT_PATH_KMAX)
    parser.add_argument(
        "--path-points-per-side", type=int, default=DEFAULT_PATH_POINTS_PER_SIDE
    )
    parser.add_argument(
        "--near-gamma-points-per-side",
        type=int,
        default=DEFAULT_NEAR_GAMMA_POINTS_PER_SIDE,
        help="Short-path point count per side used only in coarse/refine near-Gamma scoring.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Base directory where a timestamped project folder will be created.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=DEFAULT_PROJECT_NAME,
        help="Project name used in titles and output folder naming.",
    )
    parser.add_argument(
        "--fast-test",
        action="store_true",
        help="Use a lightweight profile for quick scouting before formal verification.",
    )
    parser.add_argument(
        "--no-show-plots",
        action="store_true",
        help="If set, only save figures to disk and do not open interactive plot windows.",
    )
    return parser.parse_args()


def apply_fast_test_profile(args: argparse.Namespace) -> List[str]:
    changes: List[str] = []
    fast_limits = {
        "a_points": 3,
        "r_points": 3,
        "d_points": 4,
        "a_expand_rounds": 2,
        "coarse_bootstrap_points": 12,
        "bo_calls": 0,
        "bo_random_starts": 0,
        "parallel_chunk_size": 1,
        "refine_factor": 1,
        "refine_bo_calls_per_center": 2,
        "refine_bo_random_starts": 1,
        "purcell_top_evals": 3,
        "purcell_xy_samples": 31,
        "purcell_z_samples": 7,
        "robust_top_candidates": 2,
        "robust_samples": 4,
        "max_refine_centers": 3,
        "max_near_gamma_evals": 6,
        "near_gamma_points_per_side": 3,
        "path_points_per_side": 11,
        "numeig_gamma": 8,
        "numeig_path": 8,
    }
    fast_float_limits = {
        "gmax": 3.8,
        "validate_gmax": 4.8,
    }

    for key, limit in fast_limits.items():
        current = int(getattr(args, key))
        if current != limit:
            setattr(args, key, min(current, int(limit)) if limit > 0 else int(limit))
            changes.append(f"{key}={getattr(args, key)}")

    for key, limit in fast_float_limits.items():
        current = float(getattr(args, key))
        if current > limit:
            setattr(args, key, limit)
            changes.append(f"{key}={limit}")

    return changes


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------


def resolve_parallel_workers(args: argparse.Namespace) -> int:
    # Choose a safe default worker count for Windows laptops.
    requested = int(args.workers)
    if requested > 0:
        return requested
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, max(1, cpu_count // 2)))


def resolve_parallel_chunk_size(args: argparse.Namespace, total_tasks: int) -> int:
    # Keep process-pool submission overhead under control.
    requested = int(args.parallel_chunk_size)
    if requested > 0:
        return max(1, requested)
    workers = max(1, int(getattr(args, "parallel_workers", 1)))
    return max(1, math.ceil(total_tasks / max(1, 4 * workers)))


def configure_parallel_env(threads_per_worker: int) -> None:
    # Limit nested BLAS/OpenMP threading before spawning worker processes.
    thread_text = str(max(1, int(threads_per_worker)))
    for env_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[env_name] = thread_text


def use_parallel_pool(args: argparse.Namespace, total_tasks: int) -> bool:
    # Only fan out when there is enough work to amortize spawn overhead.
    return int(getattr(args, "parallel_workers", 1)) > 1 and total_tasks >= 2


def detect_acceleration() -> Dict[str, bool]:
    # Lightweight check for optional acceleration packages in the environment.
    return {
        "cupy": importlib.util.find_spec("cupy") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }


def configure_matplotlib(show_plots: bool) -> None:
    # Prefer interactive backends when available, otherwise fall back to Agg.
    if not show_plots:
        plt.switch_backend("Agg")
    else:
        for backend in ("TkAgg", "QtAgg"):
            try:
                plt.switch_backend(backend)
                break
            except Exception:
                continue
        else:
            # Always keep file output available even without an interactive backend.
            plt.switch_backend("Agg")

    if HAS_SCIENCEPLOTS:
        plt.style.use(["science", "nature", "no-latex"])
    else:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 16,
            "axes.titleweight": "semibold",
            "figure.titlesize": 18,
            "legend.fontsize": 17,
            "legend.title_fontsize": 17,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.default": "regular",
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
        }
    )


def sanitize_project_name(project_name: str) -> str:
    # Keep output directory names filesystem-safe.
    safe = []
    for ch in project_name.strip():
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    cleaned = "".join(safe).strip("_")
    return cleaned or DEFAULT_PROJECT_NAME


def parse_r_values_over_a(text: str) -> np.ndarray:
    values = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("r-values-over-a must contain at least one value")
    return np.array(sorted(set(values)), dtype=float)


def build_r_values_over_a(args: argparse.Namespace) -> np.ndarray:
    # Build scanned r/a values from a range or an optional legacy list.
    if args.r_values_over_a:
        return parse_r_values_over_a(args.r_values_over_a)
    if args.r_points < 1:
        raise ValueError("r_points >= 1 is required")
    if args.r_max_over_a < args.r_min_over_a:
        raise ValueError("r_max_over_a must be >= r_min_over_a")
    return np.linspace(float(args.r_min_over_a), float(args.r_max_over_a), int(args.r_points))


def format_r_scan_description(args: argparse.Namespace) -> str:
    # Format hole radius ratio settings for text output.
    if args.r_values_over_a:
        return args.r_values_over_a
    return f"[{args.r_min_over_a:.4f}, {args.r_max_over_a:.4f}], {args.r_points} points"


def describe_scan_mode(args: argparse.Namespace) -> str:
    # Format the active scan mode in plain language.
    if args.scan_mode == "a_d":
        return (
            "Scan mode a_d | scanning period a and thickness ratio d/a, "
            f"fixed hole radius ratio r/a = {args.fixed_r_over_a:.4f}"
        )
    if args.scan_mode == "a_r":
        return (
            "Scan mode a_r | scanning period a and hole radius ratio r/a, "
            f"fixed thickness ratio d/a = {args.fixed_d_over_a:.4f}"
        )
    if args.scan_mode == "r_d":
        return (
            "Scan mode r_d | scanning hole radius ratio r/a and thickness ratio d/a, "
            f"fixed period a = {args.fixed_a_nm:.3f} nm"
        )
    return "Scan mode a_r_d | scanning period a, hole radius ratio r/a, and thickness ratio d/a"


def report_progress(message: str, progress_log: Optional[List[str]] = None) -> None:
    # Print progress live and also keep it for the summary report.
    print(message)
    if progress_log is not None:
        progress_log.append(message)


def record_stage_time(
    stage_times: Dict[str, float],
    stage_key: str,
    label: str,
    elapsed: float,
    progress_log: Optional[List[str]] = None,
) -> None:
    # Store per-stage timing and emit one consistent log line.
    stage_times[stage_key] = elapsed
    report_progress(f"{label} elapsed = {elapsed:.2f} s", progress_log)


def make_timestamped_output_dir(base_outdir: str, project_name: str) -> str:
    # Create a fresh timestamped output folder for each run.
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{sanitize_project_name(project_name)}"
    return os.path.abspath(os.path.join(base_outdir, folder_name))


# -----------------------------------------------------------------------------
# Geometry and Gamma eval
# -----------------------------------------------------------------------------


def make_phc(d_over_a: float, n_slab: float, n_hole: float, r_over_a: float):
    # Build the square-lattice slab for the requested normalized thickness.
    lattice = legume.Lattice("square")
    phc = legume.PhotCryst(lattice, eps_l=n_hole**2, eps_u=n_hole**2)
    phc.add_layer(d=d_over_a, eps_b=n_slab**2)
    phc.layers[-1].add_shape(
        legume.Circle(eps=n_hole**2, r=r_over_a, x_cent=0.0, y_cent=0.0)
    )
    return lattice, phc


def evaluate_scan_point_gamma(
    point: ScanPoint,
    family: FamilyConfig,
    args: argparse.Namespace,
) -> ScanPointResult:
    # Solve the Gamma point for one geometry and collect BIC-related metrics.
    _, phc = make_phc(point.d_over_a, args.n_slab, args.n_hole, point.r_over_a)
    gme = legume.GuidedModeExp(phc, gmax=args.gmax, truncate_g="abs")
    gme.run(
        kpoints=np.array([[0.0], [0.0]]),
        gmode_inds=list(family.gmode_inds),
        numeig=args.numeig_gamma,
        compute_im=True,
        gradients="approx",
        verbose=False,
    )

    freqs = np.asarray(gme.freqs).ravel()
    freqs_im = np.asarray(gme.freqs_im).ravel()
    q_vals = freqs / (2.0 * np.abs(freqs_im) + 1e-16)
    lambda_nm = np.divide(
        point.a_nm,
        freqs,
        out=np.full_like(freqs, np.nan, dtype=float),
        where=np.abs(freqs) > 1e-12,
    )

    window_mask = (
        np.isfinite(lambda_nm)
        & (lambda_nm >= args.range1_nm)
        & (lambda_nm <= args.range2_nm)
    )
    window_mode_count = int(np.sum(window_mask))
    valid_mask = (
        window_mask
        & (np.abs(freqs_im) <= args.im_threshold)
        & (q_vals >= args.q_threshold)
    )
    bic_count = int(np.sum(valid_mask))

    if np.any(window_mask):
        window_indices = np.where(window_mask)[0]
        # Keep one representative in-window mode even before the hard BIC test passes.
        score_arr = (
            np.array([loss_score_from_im(v) for v in freqs_im[window_indices]])
            + 0.2 * np.log10(np.clip(q_vals[window_indices], 1.0, 1e30))
            - 0.05 * np.abs(lambda_nm[window_indices] - 0.5 * (args.range1_nm + args.range2_nm))
        )
        best_idx = int(window_indices[np.argmax(score_arr)])
        best_freq = float(freqs[best_idx])
        best_freq_im = float(freqs_im[best_idx])
        best_q = float(q_vals[best_idx])
        best_lambda_nm = float(lambda_nm[best_idx])
    else:
        target_lambda = 0.5 * (args.range1_nm + args.range2_nm)
        finite_indices = np.where(np.isfinite(lambda_nm))[0]
        if finite_indices.size > 0:
            best_idx = int(finite_indices[np.argmin(np.abs(lambda_nm[finite_indices] - target_lambda))])
            best_freq = float(freqs[best_idx])
            best_freq_im = float(freqs_im[best_idx])
            best_q = float(q_vals[best_idx])
            best_lambda_nm = float(lambda_nm[best_idx])
        else:
            best_idx = -1
            best_freq = math.nan
            best_freq_im = math.nan
            best_q = math.nan
            best_lambda_nm = math.nan

    return ScanPointResult(
        family_key=family.key,
        a_nm=point.a_nm,
        r_over_a=point.r_over_a,
        h_nm=point.h_nm,
        d_over_a=point.d_over_a,
        window_mode_count=window_mode_count,
        bic_count=bic_count,
        best_band_index=best_idx,
        best_freq=best_freq,
        best_freq_im=best_freq_im,
        best_q=best_q,
        best_lambda_nm=best_lambda_nm,
        near_gamma_score=-1.0,
    )


def score_terms_from_result(result: ScanPointResult, args: argparse.Namespace) -> Dict[str, float]:
    # Expose the nominal objective terms so they can be plotted and audited.
    target_lambda = 0.5 * (args.range1_nm + args.range2_nm)
    if not np.isfinite(result.best_lambda_nm):
        return {
            "loss_term": -50.0,
            "q_term": -10.0,
            "thickness_term": 0.0,
            "single_mode_term": -18.0,
            "single_bic_term": -14.0,
            "lambda_penalty": 1e3,
            "continuous_score": -1e3,
        }

    lambda_penalty = 0.08 * abs(result.best_lambda_nm - target_lambda)
    loss_term = loss_score_from_im(result.best_freq_im) if np.isfinite(result.best_freq_im) else -50.0
    q_term = 0.25 * math.log10(max(result.best_q, 1.0)) if np.isfinite(result.best_q) else -10.0
    thickness_term = -2.0 * abs(result.d_over_a - (1.0 / 3.0))
    single_mode_term = 70.0 if result.window_mode_count == 1 else -18.0 * abs(result.window_mode_count - 1)
    single_bic_term = 40.0 if result.bic_count == 1 else -14.0 * abs(result.bic_count - 1)
    continuous_score = single_mode_term + single_bic_term + loss_term + q_term + thickness_term - lambda_penalty
    return {
        "loss_term": float(loss_term),
        "q_term": float(q_term),
        "thickness_term": float(thickness_term),
        "single_mode_term": float(single_mode_term),
        "single_bic_term": float(single_bic_term),
        "lambda_penalty": float(lambda_penalty),
        "continuous_score": float(continuous_score),
    }


def continuous_score_from_result(result: ScanPointResult, args: argparse.Namespace) -> float:
    # Continuous ranking objective for BO and candidate sorting.
    return float(score_terms_from_result(result, args)["continuous_score"])


def purcell_rank_bonus(purcell_factor_est: float, args: argparse.Namespace) -> float:
    # Keep Purcell as a bounded tie-breaker instead of the primary screening objective.
    return 0.0


def total_score_from_result(result: ScanPointResult, args: argparse.Namespace) -> float:
    # Unified score used by BO and candidate ranking.
    # Purcell is intentionally excluded so it remains a pure reporting quantity.
    return (
        continuous_score_from_result(result, args)
        + max(float(result.near_gamma_score), 0.0)
    )


def _integrate_xy_unit_cell(values: np.ndarray, xgrid: np.ndarray, ygrid: np.ndarray) -> float:
    # Integrate one xy slice over a single unit cell.
    arr = np.asarray(values, dtype=float)
    return float(np.trapz(np.trapz(arr, ygrid, axis=1), xgrid, axis=0))


def effective_index_from_gme(gme: legume.GuidedModeExp) -> float:
    # Use Legume's internal effective-layer permittivity used by the guided-mode basis.
    eps_layers = np.real(np.asarray(gme.eps_array, dtype=float))[1:-1]
    d_layers = np.real(np.asarray(gme.d_array, dtype=float))
    if eps_layers.size == 0:
        return 1.0
    if d_layers.size == eps_layers.size and np.sum(d_layers) > 0:
        eps_eff = float(np.average(eps_layers, weights=d_layers))
    else:
        eps_eff = float(np.mean(eps_layers))
    return math.sqrt(max(eps_eff, 1e-12))


def estimate_purcell_for_result(
    result: ScanPointResult,
    family: FamilyConfig,
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    # Approximate unit-cell Purcell factor using only the periodic slab volume.
    # The Purcell formula is evaluated explicitly in the labeled units:
    #   Fp = (3 / 4pi^2) * (lambda_m / n_eff)^3 * Q / Veff_m^3
    if result.best_band_index < 0 or not np.isfinite(result.d_over_a) or result.d_over_a <= 0:
        return math.nan, math.nan, math.nan

    _, phc = make_phc(result.d_over_a, args.n_slab, args.n_hole, result.r_over_a)
    gme = legume.GuidedModeExp(phc, gmax=args.validate_gmax, truncate_g="abs")
    gme.run(
        kpoints=np.array([[0.0], [0.0]]),
        gmode_inds=list(family.gmode_inds),
        numeig=max(args.numeig_gamma, int(result.best_band_index) + 2),
        compute_im=True,
        gradients="approx",
        verbose=False,
    )

    freqs = np.asarray(gme.freqs)
    freqs_im = np.asarray(gme.freqs_im)
    if freqs.ndim == 2:
        freq_line = freqs[0]
        freq_im_line = freqs_im[0]
    else:
        freq_line = freqs.ravel()
        freq_im_line = freqs_im.ravel()

    band_index = int(result.best_band_index)
    if band_index >= freq_line.size:
        return math.nan, math.nan, math.nan

    freq_val = float(freq_line[band_index])
    freq_im_val = float(freq_im_line[band_index])
    if not np.isfinite(freq_val) or abs(freq_val) <= 1e-12:
        return math.nan, math.nan, math.nan

    q_val = float(freq_val / (2.0 * abs(freq_im_val) + 1e-16))
    xgrid: Optional[np.ndarray] = None
    ygrid: Optional[np.ndarray] = None
    z_count = max(3, int(args.purcell_z_samples))
    z_values = np.linspace(0.0, float(result.d_over_a), z_count)
    slice_integrals: List[float] = []
    peak_density = 0.0

    for z_val in z_values:
        field_dict, xgrid, ygrid = gme.get_field_xy(
            "E",
            kind=0,
            mind=band_index,
            z=float(z_val),
            xgrid=xgrid,
            ygrid=ygrid,
            component="xyz",
            Nx=max(21, int(args.purcell_xy_samples)),
            Ny=max(21, int(args.purcell_xy_samples)),
        )
        ex = np.asarray(field_dict["x"])
        ey = np.asarray(field_dict["y"])
        ez = np.asarray(field_dict["z"])
        e_sq = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
        eps_xy, _, _ = gme.get_eps_xy(z=float(z_val), xgrid=xgrid, ygrid=ygrid)
        energy_density = np.real(np.asarray(eps_xy)) * np.real(e_sq)
        slice_integrals.append(_integrate_xy_unit_cell(energy_density, xgrid, ygrid))
        if energy_density.size > 0:
            peak_density = max(peak_density, float(np.nanmax(energy_density)))

    if peak_density <= 0 or not slice_integrals:
        return math.nan, math.nan, math.nan

    if len(z_values) > 1:
        total_energy = float(np.trapz(np.asarray(slice_integrals, dtype=float), z_values))
    else:
        total_energy = float(slice_integrals[0] * result.d_over_a)
    if total_energy <= 0:
        return math.nan, math.nan, math.nan

    mode_volume_a3 = total_energy / peak_density
    mode_volume_m3 = mode_volume_a3 * (float(result.a_nm) * 1e-9) ** 3
    n_eff = effective_index_from_gme(gme)
    lambda_m = float(result.best_lambda_nm) * 1e-9
    if not np.isfinite(lambda_m) or lambda_m <= 0 or mode_volume_m3 <= 0:
        return math.nan, float(mode_volume_a3), math.nan
    purcell_factor = (
        (3.0 / (4.0 * math.pi**2))
        * q_val
        * (lambda_m / max(n_eff, 1e-12)) ** 3
        / max(mode_volume_m3, 1e-30)
    )
    return float(purcell_factor), float(mode_volume_a3), float(mode_volume_m3)


def enrich_results_with_purcell(
    scan_results: Sequence[ScanPointResult],
    family: FamilyConfig,
    args: argparse.Namespace,
    max_evals: int,
    progress_log: Optional[List[str]] = None,
    stage_label: str = "purcell",
) -> List[ScanPointResult]:
    # Compute the slab-confined unit-cell Purcell estimate only for a small shortlist.
    if not scan_results or max_evals < 1:
        return list(scan_results)

    indexed = list(enumerate(scan_results))
    indexed.sort(
        key=lambda item: (
            item[1].window_mode_count == 1,
            item[1].bic_count == 1,
            total_score_from_result(item[1], args),
        ),
        reverse=True,
    )
    target_indices = [
        idx
        for idx, item in indexed
        if item.best_band_index >= 0 and not np.isfinite(item.purcell_factor_est)
    ][: max(1, int(max_evals))]

    if not target_indices:
        return list(scan_results)

    updated = list(scan_results)
    for local_rank, idx in enumerate(target_indices, start=1):
        item = updated[idx]
        report_progress(
            (
                f"{stage_label} {local_rank}/{len(target_indices)} | a = {item.a_nm:.3f} nm, "
                f"hole radius ratio r/a = {item.r_over_a:.4f}, h = {item.h_nm:.3f} nm"
            ),
            progress_log,
        )
        purcell_factor, mode_volume_a3, mode_volume_m3 = estimate_purcell_for_result(item, family, args)
        updated[idx] = replace(
            item,
            purcell_factor_est=purcell_factor,
            mode_volume_a3=mode_volume_a3,
            mode_volume_m3=mode_volume_m3,
        )
    return updated


def score_scan_point_near_gamma(
    point: ScanPoint,
    family: FamilyConfig,
    args: argparse.Namespace,
    path_kmax: float,
    points_per_side: int,
    band_index: int,
) -> float:
    # Recheck a short near-Gamma path for a candidate that already looks good at Gamma.
    if band_index < 0:
        return -1.0

    _, phc = make_phc(point.d_over_a, args.n_slab, args.n_hole, point.r_over_a)
    _, kpoints = build_signed_gamma_path(path_kmax, points_per_side)
    gamma_index = points_per_side
    left_slice = slice(0, gamma_index)
    right_slice = slice(gamma_index + 1, None)

    gme = legume.GuidedModeExp(phc, gmax=args.gmax, truncate_g="abs")
    gme.run(
        kpoints=kpoints,
        gmode_inds=list(family.gmode_inds),
        numeig=max(args.numeig_gamma, band_index + 2),
        compute_im=True,
        gradients="approx",
        verbose=False,
    )
    freqs = np.asarray(gme.freqs)
    freqs_im = np.asarray(gme.freqs_im)
    q_vals = freqs / (2.0 * np.abs(freqs_im) + 1e-16)
    lambda_nm = np.divide(
        point.a_nm,
        freqs,
        out=np.full_like(freqs, np.nan, dtype=float),
        where=np.abs(freqs) > 1e-12,
    )

    if band_index >= freqs.shape[1]:
        return -1.0
    gamma_lambda = float(lambda_nm[gamma_index, band_index])
    gamma_loss = float(abs(freqs_im[gamma_index, band_index]))
    gamma_q = float(q_vals[gamma_index, band_index])
    if not (args.range1_nm <= gamma_lambda <= args.range2_nm):
        return -1.0
    if gamma_loss > args.im_threshold or gamma_q < args.q_threshold:
        return -1.0

    left_loss = float(np.nanmax(np.abs(freqs_im[left_slice, band_index]))) if gamma_index > 0 else math.inf
    right_loss = float(np.nanmax(np.abs(freqs_im[right_slice, band_index]))) if freqs_im.shape[0] > gamma_index + 1 else math.inf
    left_q = float(np.nanmin(q_vals[left_slice, band_index])) if gamma_index > 0 else 0.0
    right_q = float(np.nanmin(q_vals[right_slice, band_index])) if freqs_im.shape[0] > gamma_index + 1 else 0.0

    if left_loss > args.im_threshold or right_loss > args.im_threshold:
        return -1.0
    if left_q < args.q_threshold or right_q < args.q_threshold:
        return -1.0

    # Reward both lower loss and higher Q away from Gamma.
    return loss_score_from_im(max(left_loss, right_loss)) + 0.25 * math.log10(max(min(left_q, right_q), 1.0))


def loss_score_from_im(freq_im: float) -> float:
    # Smaller imag(freq) means lower radiative loss, so larger score is better.
    loss = max(abs(float(freq_im)), 1e-30)
    return -math.log10(loss)


def evaluate_scan_chunk_worker(
    chunk: Sequence[Tuple[int, ScanPoint]],
    family: FamilyConfig,
    args: argparse.Namespace,
) -> List[Tuple[int, ScanPointResult]]:
    # Run one explicit scan chunk inside a worker process.
    output: List[Tuple[int, ScanPointResult]] = []
    for idx, point in chunk:
        output.append((idx, evaluate_scan_point_gamma(point, family, args)))
    return output


def evaluate_near_gamma_worker(
    item: ScanPointResult,
    family: FamilyConfig,
    args: argparse.Namespace,
    path_kmax: float,
    points_per_side: int,
) -> Tuple[Tuple[float, float, float], float]:
    # Run one near-Gamma score inside a worker process.
    point = ScanPoint(item.a_nm, item.r_over_a, item.h_nm, item.d_over_a)
    score = score_scan_point_near_gamma(
        point,
        family,
        args,
        path_kmax,
        points_per_side,
        item.best_band_index,
    )
    return (item.a_nm, item.r_over_a, item.d_over_a), score


def make_scan_key(a_nm: float, r_over_a: float, d_over_a: float) -> Tuple[float, float, float]:
    # Build a stable geometry key for deduplication.
    return (round(float(a_nm), 9), round(float(r_over_a), 9), round(float(d_over_a), 9))


def passes_target_constraints(result: ScanPointResult, args: argparse.Namespace) -> bool:
    # Treat one in-window, low-loss, high-Q mode as a robust "pass".
    return bool(
        result.best_band_index >= 0
        and result.window_mode_count == 1
        and result.bic_count >= 1
        and np.isfinite(result.best_lambda_nm)
        and args.range1_nm <= result.best_lambda_nm <= args.range2_nm
        and np.isfinite(result.best_freq_im)
        and abs(result.best_freq_im) <= args.im_threshold
        and np.isfinite(result.best_q)
        and result.best_q >= args.q_threshold
    )


def sample_fabrication_variant(
    candidate: ScanPointResult,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Optional[ScanPoint]:
    # Perturb physical dimensions in nm, then convert back to the normalized geometry.
    a_nm = float(candidate.a_nm + rng.normal(0.0, max(float(args.robust_a_sigma_nm), 0.0)))
    radius_nm = float(
        candidate.a_nm * candidate.r_over_a
        + rng.normal(0.0, max(float(args.robust_radius_sigma_nm), 0.0))
    )
    h_nm = float(candidate.h_nm + rng.normal(0.0, max(float(args.robust_h_sigma_nm), 0.0)))

    if not (np.isfinite(a_nm) and np.isfinite(radius_nm) and np.isfinite(h_nm)):
        return None
    if a_nm <= 1e-9 or radius_nm <= 0.0 or h_nm <= 0.0 or h_nm < float(args.min_h_nm):
        return None
    if radius_nm >= 0.49 * a_nm:
        return None

    r_over_a = radius_nm / a_nm
    d_over_a = h_nm / a_nm
    if r_over_a <= 0.0 or d_over_a <= 0.0:
        return None

    return ScanPoint(
        a_nm=a_nm,
        r_over_a=r_over_a,
        h_nm=h_nm,
        d_over_a=d_over_a,
    )


def evaluate_candidate_robustness(
    candidate: ScanPointResult,
    rank: int,
    family: FamilyConfig,
    args: argparse.Namespace,
    rng: np.random.Generator,
    progress_log: Optional[List[str]] = None,
) -> RobustCandidateSummary:
    # Monte Carlo fabrication-tolerance screen around one nominally good design.
    total_samples = max(1, int(args.robust_samples))
    scores: List[float] = []
    passing_samples = 0
    valid_samples = 0

    for sample_idx in range(total_samples):
        point = sample_fabrication_variant(candidate, args, rng)
        if point is None:
            continue
        valid_samples += 1
        result = evaluate_scan_point_gamma(point, family, args)
        score = continuous_score_from_result(result, args)
        if np.isfinite(score):
            scores.append(float(score))
        if passes_target_constraints(result, args):
            passing_samples += 1
        if progress_log is not None and ((sample_idx + 1) == total_samples or (sample_idx + 1) % 5 == 0):
            report_progress(
                (
                    f"robustness candidate {rank} sample {sample_idx + 1}/{total_samples} | "
                    f"valid = {valid_samples}, pass = {passing_samples}"
                ),
                progress_log,
            )

    nominal_score = float(continuous_score_from_result(candidate, args))
    if scores:
        score_arr = np.asarray(scores, dtype=float)
        mean_score = float(np.mean(score_arr))
        std_score = float(np.std(score_arr))
        score_q10 = float(np.quantile(score_arr, 0.10))
        robust_score = float(mean_score - float(args.robust_beta) * std_score)
    else:
        mean_score = math.nan
        std_score = math.nan
        score_q10 = math.nan
        robust_score = -1e9

    summary = RobustCandidateSummary(
        rank=rank,
        a_nm=float(candidate.a_nm),
        r_over_a=float(candidate.r_over_a),
        h_nm=float(candidate.h_nm),
        d_over_a=float(candidate.d_over_a),
        nominal_score=nominal_score,
        mean_score=mean_score,
        std_score=std_score,
        robust_score=robust_score,
        score_q10=score_q10,
        yield_fraction=float(passing_samples / total_samples),
        valid_fraction=float(valid_samples / total_samples),
        passing_samples=passing_samples,
        valid_samples=valid_samples,
        total_samples=total_samples,
    )
    report_progress(
        (
            f"Robustness candidate {rank} done | yield = {summary.yield_fraction:.3f}, "
            f"robust score = {summary.robust_score:.3f}, q10 = {summary.score_q10:.3f}"
        ),
        progress_log,
    )
    return summary


def run_fabrication_robustness_screen(
    candidates: Sequence[ScanPointResult],
    family: FamilyConfig,
    args: argparse.Namespace,
    progress_log: Optional[List[str]] = None,
) -> Tuple[List[RobustCandidateSummary], Optional[RobustCandidateSummary], float]:
    # Apply a small Monte Carlo post-check to the final shortlist.
    t0 = time.time()
    top_count = max(0, int(args.robust_top_candidates))
    sample_count = max(0, int(args.robust_samples))
    if top_count < 1 or sample_count < 1 or not candidates:
        return [], None, 0.0

    shortlist = select_top_scan_candidates(
        candidates,
        max_candidates=top_count,
        args=args,
    )
    if not shortlist:
        return [], None, time.time() - t0

    report_progress(
        (
            f"Fabrication robustness screen | top {len(shortlist)} candidates, "
            f"{sample_count} samples each, sigmas = "
            f"(a {args.robust_a_sigma_nm:.2f} nm, r {args.robust_radius_sigma_nm:.2f} nm, "
            f"h {args.robust_h_sigma_nm:.2f} nm)"
        ),
        progress_log,
    )

    rng = np.random.default_rng(int(args.robust_seed))
    summaries: List[RobustCandidateSummary] = []
    for rank, candidate in enumerate(shortlist, start=1):
        report_progress(
            (
                f"Robustness candidate {rank}/{len(shortlist)} | a = {candidate.a_nm:.3f} nm, "
                f"hole radius ratio r/a = {candidate.r_over_a:.4f}, h = {candidate.h_nm:.3f} nm"
            ),
            progress_log,
        )
        summaries.append(
            evaluate_candidate_robustness(
                candidate,
                rank,
                family,
                args,
                rng,
                progress_log=progress_log,
            )
        )

    best = max(
        summaries,
        key=lambda item: (
            item.yield_fraction,
            item.robust_score,
            item.score_q10 if np.isfinite(item.score_q10) else -1e9,
            item.nominal_score,
        ),
    ) if summaries else None
    return summaries, best, time.time() - t0


# -----------------------------------------------------------------------------
# Scan axis helpers
# -----------------------------------------------------------------------------


def build_scan_axes(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Build the scan axes for the requested mode and window.
    return build_scan_axes_for_window(args)


def build_scan_axes_for_window(
    args: argparse.Namespace,
    a_min_nm: Optional[float] = None,
    a_max_nm: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = args.scan_mode
    if mode == "r_d":
        a_values = np.array([float(args.fixed_a_nm)], dtype=float)
    else:
        local_a_min = a_min_nm if a_min_nm is not None else (args.a_min_nm if args.a_min_nm is not None else 0.1 * args.range1_nm)
        local_a_max = a_max_nm if a_max_nm is not None else (args.a_max_nm if args.a_max_nm is not None else args.range1_nm - 1e-6)
        if args.allow_superwavelength and args.a_max_nm is None and a_max_nm is None:
            local_a_max = args.range2_nm
        a_values = np.linspace(float(local_a_min), float(local_a_max), args.a_points)

    if mode in ("a_r_d", "a_r", "r_d"):
        r_values = build_r_values_over_a(args)
    else:
        r_values = np.array([float(args.fixed_r_over_a)], dtype=float)

    if mode in ("a_r_d", "a_d", "r_d"):
        d_values = np.linspace(args.d_min, args.d_max, args.d_points)
    else:
        d_values = np.array([float(args.fixed_d_over_a)], dtype=float)

    return a_values, r_values, d_values


# -----------------------------------------------------------------------------
# Grid scan and scoring
# -----------------------------------------------------------------------------


def build_bo_dimensions(
    a_values: np.ndarray,
    r_values: np.ndarray,
    d_values: np.ndarray,
    args: argparse.Namespace,
) -> List[object]:
    # Keep BO dimension definitions consistent with the active scan mode.
    dimensions: List[object] = []
    if args.scan_mode == "r_d":
        dimensions.append(Categorical([float(args.fixed_a_nm)], name="a_nm"))
    elif len(a_values) <= 1:
        dimensions.append(Categorical([float(a_values[0])], name="a_nm"))
    else:
        dimensions.append(Real(float(a_values[0]), float(a_values[-1]), name="a_nm"))

    if args.scan_mode in ("a_r_d", "a_r", "r_d") and len(r_values) > 1:
        dimensions.append(Categorical([float(v) for v in r_values], name="r_over_a"))
    else:
        fixed_r = float(r_values[0]) if len(r_values) else float(args.fixed_r_over_a)
        dimensions.append(Categorical([fixed_r], name="r_over_a"))

    if args.scan_mode in ("a_r_d", "a_d", "r_d") and len(d_values) > 1:
        dimensions.append(Real(float(d_values[0]), float(d_values[-1]), name="d_over_a"))
    else:
        fixed_d = float(d_values[0]) if len(d_values) else float(args.fixed_d_over_a)
        dimensions.append(Categorical([fixed_d], name="d_over_a"))

    return dimensions


def build_adaptive_a_windows(args: argparse.Namespace) -> List[Tuple[float, float]]:
    # Start near half of the target wavelength and expand outward if needed.
    if args.scan_mode == "r_d":
        return [(float(args.fixed_a_nm), float(args.fixed_a_nm))]

    hard_min = args.a_min_nm if args.a_min_nm is not None else 0.1 * args.range1_nm
    if args.a_max_nm is not None:
        hard_max = args.a_max_nm
    elif args.allow_superwavelength:
        hard_max = args.range2_nm
    else:
        hard_max = args.range1_nm - 1e-6

    start_min = max(hard_min, 0.5 * args.range1_nm)
    start_max = min(hard_max, 0.5 * args.range2_nm)
    if start_max <= start_min:
        start_min, start_max = hard_min, hard_max

    windows: List[Tuple[float, float]] = []
    full_span = hard_max - hard_min
    base_span = start_max - start_min
    if base_span <= 0:
        return [(hard_min, hard_max)]

    center = 0.5 * (start_min + start_max)
    for round_idx in range(max(1, args.a_expand_rounds)):
        frac = round_idx / max(1, args.a_expand_rounds - 1) if args.a_expand_rounds > 1 else 1.0
        span = base_span + frac * (full_span - base_span)
        a_min = max(hard_min, center - 0.5 * span)
        a_max = min(hard_max, center + 0.5 * span)
        window = (float(a_min), float(a_max))
        if not windows or windows[-1] != window:
            windows.append(window)

    if not windows or windows[-1] != (float(hard_min), float(hard_max)):
        windows.append((float(hard_min), float(hard_max)))
    return windows


def select_evenly_spaced_values(values: np.ndarray, max_count: int) -> np.ndarray:
    # Downsample one axis while keeping endpoints and broad coverage.
    if len(values) <= max_count:
        return values
    indices = np.unique(np.round(np.linspace(0, len(values) - 1, max_count)).astype(int))
    return values[indices]


def build_coarse_bootstrap_axes(
    a_values: np.ndarray,
    r_values: np.ndarray,
    d_values: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Build a sparse bootstrap grid so BO can take over earlier.
    max_points = max(1, int(args.coarse_bootstrap_points))
    full_product = len(a_values) * len(r_values) * len(d_values)
    if full_product <= max_points:
        return a_values, r_values, d_values

    varying_dims = sum(len(axis) > 1 for axis in (a_values, r_values, d_values))
    if varying_dims == 0:
        return a_values, r_values, d_values

    base_count = max(2, int(round(max_points ** (1.0 / varying_dims))))
    coarse_a = select_evenly_spaced_values(a_values, base_count) if len(a_values) > 1 else a_values
    coarse_r = select_evenly_spaced_values(r_values, base_count) if len(r_values) > 1 else r_values
    coarse_d = select_evenly_spaced_values(d_values, base_count) if len(d_values) > 1 else d_values
    return coarse_a, coarse_r, coarse_d


def build_scan_points(
    a_values: np.ndarray,
    r_values: np.ndarray,
    d_values: np.ndarray,
    args: argparse.Namespace,
) -> List[ScanPoint]:
    # Assemble candidate geometries and discard obviously unphysical ones.
    points: List[ScanPoint] = []
    for a_nm in a_values:
        for r_over_a in r_values:
            for d_over_a in d_values:
                h_nm = float(a_nm) * float(d_over_a)
                if h_nm < args.min_h_nm:
                    continue
                points.append(
                    ScanPoint(
                        a_nm=float(a_nm),
                        r_over_a=float(r_over_a),
                        h_nm=float(h_nm),
                        d_over_a=float(d_over_a),
                    )
                )
    return points


def run_scan_point_grid(
    scan_points: Sequence[ScanPoint],
    family: FamilyConfig,
    args: argparse.Namespace,
    progress_log: Optional[List[str]] = None,
    stage_label: str = "coarse_scan",
) -> List[ScanPointResult]:
    # Evaluate every point in the explicit scan grid.
    total = len(scan_points)
    if not use_parallel_pool(args, total):
        results: List[ScanPointResult] = []
        for idx, point in enumerate(scan_points, start=1):
            report_progress(
                (
                    f"{stage_label} {idx}/{total} | a = {point.a_nm:.3f} nm, "
                    f"hole radius ratio r/a = {point.r_over_a:.4f}, "
                    f"h = {point.h_nm:.3f} nm"
                ),
                progress_log,
            )
            results.append(evaluate_scan_point_gamma(point, family, args))
        return results

    chunk_size = resolve_parallel_chunk_size(args, total)
    report_progress(
        (
            f"{stage_label} parallel | workers = {args.parallel_workers}, "
            f"threads/worker = {args.threads_per_worker}, chunk = {chunk_size}"
        ),
        progress_log,
    )

    ordered_results: List[Optional[ScanPointResult]] = [None] * total
    indexed_points = list(enumerate(scan_points, start=1))
    chunks = [
        indexed_points[start : start + chunk_size]
        for start in range(0, total, chunk_size)
    ]

    done = 0
    with ProcessPoolExecutor(
        max_workers=args.parallel_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        future_map = {
            executor.submit(evaluate_scan_chunk_worker, chunk, family, args): chunk
            for chunk in chunks
        }
        for future in as_completed(future_map):
            chunk_results = future.result()
            for idx, result in chunk_results:
                ordered_results[idx - 1] = result
            done += len(chunk_results)
            last_result = chunk_results[-1][1]
            report_progress(
                (
                    f"{stage_label} {done}/{total} done | "
                    f"last a = {last_result.a_nm:.3f} nm, "
                    f"hole radius ratio r/a = {last_result.r_over_a:.4f}, "
                    f"h = {last_result.h_nm:.3f} nm"
                ),
                progress_log,
            )

    return [item for item in ordered_results if item is not None]


def score_scan_results_near_gamma(
    scan_results: Sequence[ScanPointResult],
    family: FamilyConfig,
    args: argparse.Namespace,
    path_kmax: float,
    points_per_side: int,
    progress_log: Optional[List[str]] = None,
    stage_label: str = "coarse_near_gamma",
) -> List[ScanPointResult]:
    # Score only the most promising Gamma-point candidates on a short path.
    updated: List[ScanPointResult] = []
    candidates = [item for item in scan_results if item.bic_count == 1 and item.best_band_index >= 0]
    candidates = sorted(
        candidates,
        key=lambda item: (
            continuous_score_from_result(item, args),
            -abs(item.d_over_a - (1.0 / 3.0)),
        ),
        reverse=True,
    )
    max_evals = max(1, int(args.max_near_gamma_evals))
    skipped = max(0, len(candidates) - max_evals)
    candidates = candidates[:max_evals]
    total = len(candidates)
    scored_map: Dict[Tuple[float, float, float], float] = {}

    if skipped > 0:
        report_progress(
            f"{stage_label} only evaluates the top {total} candidates near Gamma and skips {skipped} others",
            progress_log,
        )

    if use_parallel_pool(args, total):
        report_progress(
            (
                f"{stage_label} parallel | workers = {args.parallel_workers}, "
                f"threads/worker = {args.threads_per_worker}"
            ),
            progress_log,
        )
        done = 0
        with ProcessPoolExecutor(
            max_workers=args.parallel_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            future_map = {
                executor.submit(
                    evaluate_near_gamma_worker,
                    item,
                    family,
                    args,
                    path_kmax,
                    points_per_side,
                ): item
                for item in candidates
            }
            for future in as_completed(future_map):
                item = future_map[future]
                key, score = future.result()
                scored_map[key] = score
                done += 1
                report_progress(
                    (
                        f"{stage_label} {done}/{total} done | a = {item.a_nm:.3f} nm, "
                        f"hole radius ratio r/a = {item.r_over_a:.4f}, "
                        f"h = {item.h_nm:.3f} nm"
                    ),
                    progress_log,
                )
    else:
        for idx, item in enumerate(candidates, start=1):
            report_progress(
                (
                    f"{stage_label} {idx}/{total} | a = {item.a_nm:.3f} nm, "
                    f"hole radius ratio r/a = {item.r_over_a:.4f}, "
                    f"h = {item.h_nm:.3f} nm"
                ),
                progress_log,
            )
            point = ScanPoint(item.a_nm, item.r_over_a, item.h_nm, item.d_over_a)
            scored_map[(item.a_nm, item.r_over_a, item.d_over_a)] = score_scan_point_near_gamma(
                point,
                family,
                args,
                path_kmax,
                points_per_side,
                item.best_band_index,
            )

    for item in scan_results:
        key = (item.a_nm, item.r_over_a, item.d_over_a)
        updated.append(
            ScanPointResult(
                family_key=item.family_key,
                a_nm=item.a_nm,
                r_over_a=item.r_over_a,
                h_nm=item.h_nm,
                d_over_a=item.d_over_a,
                window_mode_count=item.window_mode_count,
                bic_count=item.bic_count,
                best_band_index=item.best_band_index,
                best_freq=item.best_freq,
                best_freq_im=item.best_freq_im,
                best_q=item.best_q,
                best_lambda_nm=item.best_lambda_nm,
                near_gamma_score=scored_map.get(key, item.near_gamma_score),
                purcell_factor_est=item.purcell_factor_est,
                mode_volume_a3=item.mode_volume_a3,
                mode_volume_m3=item.mode_volume_m3,
            )
        )
    return updated


def predict_bo_score(
    optimizer: Optimizer,
    x: Sequence[float],
) -> Tuple[float, float]:
    # Predict the surrogate-model mean and standard deviation of the BO score.
    if not optimizer.models:
        return math.nan, math.nan

    model = optimizer.models[-1]
    try:
        x_trans = optimizer.space.transform([list(x)])
        mean_obj, std_obj = model.predict(x_trans, return_std=True)
        return float(-mean_obj[0]), float(std_obj[0])
    except Exception:
        return math.nan, math.nan


# -----------------------------------------------------------------------------
# Bayesian optimization
# -----------------------------------------------------------------------------


def run_bayesian_optimization(
    a_window: Tuple[float, float],
    r_values: np.ndarray,
    d_values: np.ndarray,
    seed_results: Sequence[ScanPointResult],
    family: FamilyConfig,
    args: argparse.Namespace,
    progress_log: Optional[List[str]] = None,
) -> Tuple[List[ScanPointResult], List[ScanPointResult], Dict[str, np.ndarray]]:
    # Seed BO with coarse-scan results, then continue with ask/tell iterations.
    if Optimizer is None or Real is None or Categorical is None:
        raise RuntimeError(
            "scikit-optimize is not available in this Python environment. "
            "Install it with `python -m pip install scikit-optimize`."
        )
    a_values = np.array([float(a_window[0]), float(a_window[1])], dtype=float)
    optimizer = Optimizer(
        dimensions=build_bo_dimensions(a_values, r_values, d_values, args),
        base_estimator="GP",
        acq_func="EI",
        n_initial_points=max(1, args.bo_random_starts),
        random_state=0,
    )

    bo_results: List[ScanPointResult] = []
    bo_candidates: List[ScanPointResult] = []
    eval_records = {
        "iteration": [],
        "pred_score": [],
        "pred_std": [],
        "true_score": [],
        "abs_error": [],
        "covered_95": [],
    }
    seen_scores: Dict[Tuple[float, float, float], float] = {}
    shortlisted_keys: List[Tuple[float, float, float]] = []
    for item in seed_results:
        x = [item.a_nm, item.r_over_a, item.d_over_a]
        score = total_score_from_result(item, args)
        optimizer.tell(x, -score)
        seen_scores[make_scan_key(item.a_nm, item.r_over_a, item.d_over_a)] = score

    total_rounds = max(1, min(int(args.bo_rounds), int(args.bo_calls)))
    round_sizes = [args.bo_calls // total_rounds] * total_rounds
    for idx in range(args.bo_calls % total_rounds):
        round_sizes[idx] += 1

    global_step = 0
    for round_idx, round_calls in enumerate(round_sizes, start=1):
        report_progress(
            f"Bayes opt round {round_idx}/{total_rounds} | target {round_calls} evaluations",
            progress_log,
        )
        round_results: List[ScanPointResult] = []

        for _ in range(round_calls):
            global_step += 1
            x = optimizer.ask()
            a_nm, r_over_a, d_over_a = map(float, x)
            h_nm = a_nm * d_over_a
            pred_score, pred_std = predict_bo_score(optimizer, x)
            geometry_key = make_scan_key(a_nm, r_over_a, d_over_a)
            report_progress(
                (
                    f"Bayes opt {global_step}/{args.bo_calls} | a = {a_nm:.3f} nm, "
                    f"hole radius ratio r/a = {r_over_a:.4f}, "
                    f"thickness ratio d/a = {d_over_a:.4f}, "
                    f"h = {h_nm:.3f} nm"
                ),
                progress_log,
            )

            if geometry_key in seen_scores:
                optimizer.tell(x, -seen_scores[geometry_key])
                report_progress(
                    f"Bayes opt {global_step}/{args.bo_calls} | duplicate geometry, skip expensive solve",
                    progress_log,
                )
                continue

            if d_over_a < args.d_min or d_over_a > args.d_max or h_nm < args.min_h_nm:
                bad_value = 1e3
                if d_over_a < args.d_min or d_over_a > args.d_max:
                    bad_value += 100.0 * abs(d_over_a - np.clip(d_over_a, args.d_min, args.d_max))
                if h_nm < args.min_h_nm:
                    bad_value += args.min_h_nm - h_nm
                optimizer.tell(x, bad_value)
                seen_scores[geometry_key] = -bad_value
                continue

            point = ScanPoint(a_nm=a_nm, r_over_a=r_over_a, h_nm=h_nm, d_over_a=d_over_a)
            result = evaluate_scan_point_gamma(point, family, args)
            if result.bic_count == 1 and result.best_band_index >= 0:
                near_gamma_score = score_scan_point_near_gamma(
                    point,
                    family,
                    args,
                    path_kmax=0.03,
                    points_per_side=max(3, args.near_gamma_points_per_side),
                    band_index=result.best_band_index,
                )
            else:
                near_gamma_score = -1.0

            result = ScanPointResult(
                family_key=result.family_key,
                a_nm=result.a_nm,
                r_over_a=result.r_over_a,
                h_nm=result.h_nm,
                d_over_a=result.d_over_a,
                window_mode_count=result.window_mode_count,
                bic_count=result.bic_count,
                best_band_index=result.best_band_index,
                best_freq=result.best_freq,
                best_freq_im=result.best_freq_im,
                best_q=result.best_q,
                best_lambda_nm=result.best_lambda_nm,
                near_gamma_score=near_gamma_score,
            )
            bo_results.append(result)
            round_results.append(result)

            score = total_score_from_result(result, args)
            seen_scores[geometry_key] = score
            abs_error = abs(pred_score - score) if np.isfinite(pred_score) else math.nan
            covered_95 = (
                abs_error <= 1.96 * pred_std
                if np.isfinite(abs_error) and np.isfinite(pred_std)
                else math.nan
            )
            eval_records["iteration"].append(float(global_step))
            eval_records["pred_score"].append(float(pred_score))
            eval_records["pred_std"].append(float(pred_std))
            eval_records["true_score"].append(float(score))
            eval_records["abs_error"].append(float(abs_error))
            eval_records["covered_95"].append(float(covered_95) if np.isfinite(covered_95) else math.nan)
            optimizer.tell(x, -score)

        remaining_slots = max(0, args.max_refine_centers - len(bo_candidates))
        if remaining_slots > 0 and round_results:
            round_candidates = select_new_top_scan_candidates(
                round_results,
                remaining_slots,
                args,
                shortlisted_keys,
            )
            if round_candidates:
                bo_candidates.extend(round_candidates)
                shortlisted_keys.extend(
                    make_scan_key(item.a_nm, item.r_over_a, item.d_over_a)
                    for item in round_candidates
                )
                report_progress(
                    f"Bayes opt round {round_idx} added {len(round_candidates)} new candidate(s)",
                    progress_log,
                )
        if len(bo_candidates) >= args.max_refine_centers:
            report_progress(
                "Bayes opt already filled the refinement candidate budget; stop BO rounds early",
                progress_log,
            )
            break

    bo_eval = {key: np.asarray(value, dtype=float) for key, value in eval_records.items()}
    return bo_results, bo_candidates, bo_eval


# -----------------------------------------------------------------------------
# Candidate refinement
# -----------------------------------------------------------------------------


def select_top_scan_candidates(
    scan_results: Sequence[ScanPointResult],
    max_candidates: int,
    args: argparse.Namespace,
) -> List[ScanPointResult]:
    # Promote the most promising single-BIC points into the refinement stage.
    target_lambda = 0.5 * (args.range1_nm + args.range2_nm)
    valid = [item for item in scan_results if item.bic_count == 1 and item.best_band_index >= 0]
    ranked = sorted(
        valid,
        key=lambda item: (
            item.window_mode_count == 1,
            total_score_from_result(item, args),
            item.near_gamma_score,
            continuous_score_from_result(item, args),
            -abs(item.best_lambda_nm - target_lambda),
            -abs(item.d_over_a - (1.0 / 3.0)),
        ),
        reverse=True,
    )
    return ranked[:max_candidates]


def select_new_top_scan_candidates(
    scan_results: Sequence[ScanPointResult],
    max_candidates: int,
    args: argparse.Namespace,
    excluded_keys: Sequence[Tuple[float, float, float]],
) -> List[ScanPointResult]:
    # Select only candidates that were not already shortlisted earlier.
    excluded = set(excluded_keys)
    selected: List[ScanPointResult] = []
    for item in select_top_scan_candidates(scan_results, max(1, len(scan_results)), args):
        key = make_scan_key(item.a_nm, item.r_over_a, item.d_over_a)
        if key in excluded:
            continue
        selected.append(item)
        if len(selected) >= max_candidates:
            break
    return selected


def merge_unique_candidates(
    first_group: Sequence[ScanPointResult],
    second_group: Sequence[ScanPointResult],
    max_candidates: int,
) -> List[ScanPointResult]:
    # Merge candidate lists without repeating the same geometry.
    merged: List[ScanPointResult] = []
    seen: set[Tuple[float, float, float]] = set()
    for group in (first_group, second_group):
        for item in group:
            key = make_scan_key(item.a_nm, item.r_over_a, item.d_over_a)
            if key in seen:
                continue
            merged.append(item)
            seen.add(key)
            if len(merged) >= max_candidates:
                return merged
    return merged


def merge_unique_scan_results(
    first_group: Sequence[ScanPointResult],
    second_group: Sequence[ScanPointResult],
) -> List[ScanPointResult]:
    # Merge detailed scan results while keeping the first group's values on duplicates.
    merged: Dict[Tuple[float, float, float], ScanPointResult] = {}
    for group in (first_group, second_group):
        for item in group:
            key = make_scan_key(item.a_nm, item.r_over_a, item.d_over_a)
            if key not in merged:
                merged[key] = item
    return list(merged.values())


def build_refine_axes_for_center(
    item: ScanPointResult,
    coarse_a_values: np.ndarray,
    coarse_r_values: np.ndarray,
    coarse_d_values: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Build the local refinement window centered on one shortlisted geometry.
    a_step = float(np.min(np.diff(coarse_a_values))) if len(coarse_a_values) > 1 else 10.0
    d_step = float(np.min(np.diff(coarse_d_values))) if len(coarse_d_values) > 1 else 0.02

    if args.scan_mode == "r_d":
        a_vals = np.array([float(item.a_nm)], dtype=float)
    else:
        a_vals = np.linspace(
            max(float(coarse_a_values[0]), item.a_nm - a_step),
            min(float(coarse_a_values[-1]), item.a_nm + a_step),
            max(3, 2 * args.refine_factor + 1),
        )

    if args.scan_mode in ("a_r_d", "a_r", "r_d"):
        r_idx = int(np.argmin(np.abs(coarse_r_values - item.r_over_a)))
        r_lo = max(0, r_idx - 1)
        r_hi = min(len(coarse_r_values), r_idx + 2)
        r_vals = coarse_r_values[r_lo:r_hi]
    else:
        r_vals = np.array([float(item.r_over_a)], dtype=float)

    if args.scan_mode in ("a_r_d", "a_d", "r_d"):
        d_vals = np.linspace(
            max(float(coarse_d_values[0]), item.d_over_a - d_step),
            min(float(coarse_d_values[-1]), item.d_over_a + d_step),
            max(3, 2 * args.refine_factor + 1),
        )
    else:
        d_vals = np.array([float(item.d_over_a)], dtype=float)
    return a_vals, r_vals, d_vals


def build_refined_scan_points_3d(
    centers: Sequence[ScanPointResult],
    coarse_a_values: np.ndarray,
    coarse_r_values: np.ndarray,
    coarse_d_values: np.ndarray,
    args: argparse.Namespace,
) -> List[ScanPoint]:
    # Locally densify the 3D parameter space around shortlisted candidates.
    if not centers:
        return []

    points: List[ScanPoint] = []
    for item in centers:
        a_vals, r_vals, d_vals = build_refine_axes_for_center(
            item,
            coarse_a_values,
            coarse_r_values,
            coarse_d_values,
            args,
        )
        points.extend(build_scan_points(a_vals, r_vals, d_vals, args))

    dedup: Dict[Tuple[float, float, float], ScanPoint] = {}
    for point in points:
        key = (round(point.a_nm, 9), round(point.r_over_a, 9), round(point.d_over_a, 9))
        dedup[key] = point
    return list(dedup.values())


def run_local_refinement_bo(
    candidate_pool: Sequence[ScanPointResult],
    coarse_a_values: np.ndarray,
    coarse_r_values: np.ndarray,
    coarse_d_values: np.ndarray,
    family: FamilyConfig,
    args: argparse.Namespace,
    progress_log: Optional[List[str]] = None,
) -> List[ScanPointResult]:
    # Replace dense local refinement with smaller BO runs centered on shortlisted geometries.
    if not candidate_pool or int(args.refine_bo_calls_per_center) < 1:
        return []
    if Optimizer is None or Real is None or Categorical is None:
        report_progress(
            "Local refinement BO is unavailable because scikit-optimize is missing; falling back to dense refinement.",
            progress_log,
        )
        return []

    seeded = list(candidate_pool)
    refined_results: List[ScanPointResult] = list(seeded)
    per_center_calls = max(1, int(args.refine_bo_calls_per_center))
    random_starts = max(1, min(int(args.refine_bo_random_starts), per_center_calls))

    for center_idx, center in enumerate(seeded, start=1):
        a_vals, r_vals, d_vals = build_refine_axes_for_center(
            center,
            coarse_a_values,
            coarse_r_values,
            coarse_d_values,
            args,
        )
        optimizer = Optimizer(
            dimensions=build_bo_dimensions(a_vals, r_vals, d_vals, args),
            base_estimator="GP",
            acq_func="EI",
            n_initial_points=random_starts,
            random_state=center_idx - 1,
        )
        seen_scores: Dict[Tuple[float, float, float], float] = {}
        center_key = make_scan_key(center.a_nm, center.r_over_a, center.d_over_a)
        center_score = total_score_from_result(center, args)
        optimizer.tell([center.a_nm, center.r_over_a, center.d_over_a], -center_score)
        seen_scores[center_key] = center_score
        report_progress(
            (
                f"refined_bo center {center_idx}/{len(seeded)} | local window = "
                f"{len(a_vals)} x {len(r_vals)} x {len(d_vals)}"
            ),
            progress_log,
        )

        for local_idx in range(1, per_center_calls + 1):
            x = optimizer.ask()
            a_nm, r_over_a, d_over_a = map(float, x)
            h_nm = a_nm * d_over_a
            geometry_key = make_scan_key(a_nm, r_over_a, d_over_a)
            report_progress(
                (
                    f"refined_bo center {center_idx}/{len(seeded)} eval {local_idx}/{per_center_calls} | "
                    f"a = {a_nm:.3f} nm, hole radius ratio r/a = {r_over_a:.4f}, "
                    f"thickness ratio d/a = {d_over_a:.4f}, h = {h_nm:.3f} nm"
                ),
                progress_log,
            )

            if geometry_key in seen_scores:
                optimizer.tell(x, -seen_scores[geometry_key])
                continue

            if d_over_a < args.d_min or d_over_a > args.d_max or h_nm < args.min_h_nm:
                bad_value = 1e3
                if d_over_a < args.d_min or d_over_a > args.d_max:
                    bad_value += 100.0 * abs(d_over_a - np.clip(d_over_a, args.d_min, args.d_max))
                if h_nm < args.min_h_nm:
                    bad_value += args.min_h_nm - h_nm
                optimizer.tell(x, bad_value)
                seen_scores[geometry_key] = -bad_value
                continue

            point = ScanPoint(a_nm=a_nm, r_over_a=r_over_a, h_nm=h_nm, d_over_a=d_over_a)
            result = evaluate_scan_point_gamma(point, family, args)
            if result.bic_count == 1 and result.best_band_index >= 0:
                near_gamma_score = score_scan_point_near_gamma(
                    point,
                    family,
                    args,
                    path_kmax=0.03,
                    points_per_side=max(3, args.near_gamma_points_per_side),
                    band_index=result.best_band_index,
                )
            else:
                near_gamma_score = -1.0

            result = replace(result, near_gamma_score=near_gamma_score)
            refined_results.append(result)

            score = total_score_from_result(result, args)
            seen_scores[geometry_key] = score
            optimizer.tell(x, -score)

    return refined_results


# -----------------------------------------------------------------------------
# Plot data helpers
# -----------------------------------------------------------------------------


def build_slice_maps_by_r(
    scan_results: Sequence[ScanPointResult],
    a_values: np.ndarray,
    d_values: np.ndarray,
    r_values: np.ndarray,
) -> Dict[float, np.ndarray]:
    # Build a phase-map grid for each selected hole radius ratio slice.
    maps: Dict[float, np.ndarray] = {}
    index_map = {
        (round(item.a_nm, 9), round(item.r_over_a, 9), round(item.d_over_a, 9)): item.bic_count
        for item in scan_results
    }
    for r_value in r_values:
        mat = np.full((len(d_values), len(a_values)), np.nan, dtype=float)
        for hi, d_over_a in enumerate(d_values):
            for ai, a_nm in enumerate(a_values):
                key = (round(float(a_nm), 9), round(float(r_value), 9), round(float(d_over_a), 9))
                if key in index_map:
                    mat[hi, ai] = index_map[key]
        maps[float(r_value)] = mat
    return maps


def build_phase_points_by_r(
    scan_results: Sequence[ScanPointResult],
    r_values: np.ndarray,
) -> Dict[float, List[ScanPointResult]]:
    phase_points: Dict[float, List[ScanPointResult]] = {}
    for r_value in r_values:
        target = float(r_value)
        points = [
            item
            for item in scan_results
            if abs(float(item.r_over_a) - target) < 1e-9
        ]
        points.sort(key=lambda item: (item.a_nm, item.h_nm))
        phase_points[target] = points
    return phase_points


def pick_three_r_slices(r_values: np.ndarray) -> np.ndarray:
    # Pick low, middle, and high hole radius ratio slices.
    if len(r_values) <= 3:
        return r_values
    idxs = [0, len(r_values) // 2, len(r_values) - 1]
    return np.array([r_values[i] for i in idxs], dtype=float)


def pick_best_r_slice(
    scan_results: Sequence[ScanPointResult],
    r_values: np.ndarray,
    args: argparse.Namespace,
    example: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    if example is not None:
        return np.array([float(example["r_over_a"])], dtype=float)

    if not scan_results:
        return np.array([float(r_values[0])], dtype=float)

    best_by_r: Dict[float, float] = {}
    for item in scan_results:
        r_key = float(item.r_over_a)
        score = (
            item.near_gamma_score
            if item.bic_count == 1 and item.best_band_index >= 0
            else continuous_score_from_result(item, args)
        )
        best_by_r[r_key] = max(best_by_r.get(r_key, -1e18), score)

    best_r = max(best_by_r.items(), key=lambda kv: kv[1])[0]
    return np.array([best_r], dtype=float)


# -----------------------------------------------------------------------------
# Scan pipeline stages
# -----------------------------------------------------------------------------


def verify_candidate_with_path(
    result: ScanPointResult,
    family: FamilyConfig,
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, np.ndarray]]:
    # Final validation along the +/-0.07 path near Gamma.
    example = {
        "family_key": family.key,
        "pol": family.pol_label,
        "a_nm": result.a_nm,
        "r_over_a": result.r_over_a,
        "h_nm": result.h_nm,
        "d_over_a": result.d_over_a,
        "band": result.best_band_index,
        "lambda_nm": result.best_lambda_nm,
        "freq_im": result.best_freq_im,
        "gamma_q": result.best_q,
        "purcell_factor_est": result.purcell_factor_est,
        "mode_volume_a3": result.mode_volume_a3,
        "mode_volume_m3": result.mode_volume_m3,
    }
    path_result = run_example_path(example, family, result.best_band_index, args)
    band = result.best_band_index
    q_vals = path_result["q"][:, band]
    loss_vals = np.abs(path_result["freqs_im"][:, band])
    ok = bool(np.all(q_vals >= args.q_threshold) and np.all(loss_vals <= args.im_threshold))
    return ok, path_result


def coarse_scan_with_adaptive_a(
    family: FamilyConfig,
    args: argparse.Namespace,
    progress_log: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ScanPointResult], List[ScanPointResult], float]:
    # Coarse 3D scan with adaptive expansion of the a-window.
    coarse_t0 = time.time()
    a_windows = build_adaptive_a_windows(args)

    selected_a_values = np.array([])
    r_values = np.array([])
    d_values = np.array([])
    selected_window_results: List[ScanPointResult] = []
    candidate_pool: List[ScanPointResult] = []

    for round_idx, (a_min_nm, a_max_nm) in enumerate(a_windows, start=1):
        report_progress(
            f"Coarse scan window {round_idx}/{len(a_windows)} | a in [{a_min_nm:.3f}, {a_max_nm:.3f}] nm",
            progress_log,
        )
        a_values, r_values, d_values = build_scan_axes_for_window(args, a_min_nm, a_max_nm)
        full_scan_points = build_scan_points(a_values, r_values, d_values, args)
        coarse_a_values, coarse_r_values, coarse_d_values = build_coarse_bootstrap_axes(
            a_values,
            r_values,
            d_values,
            args,
        )
        scan_points = build_scan_points(coarse_a_values, coarse_r_values, coarse_d_values, args)
        report_progress(
            (
                f"Coarse bootstrap start | sampled {len(scan_points)} of {len(full_scan_points)} "
                f"(a, hole radius ratio r/a, thickness ratio d/a) points before BO"
            ),
            progress_log,
        )

        coarse_results = run_scan_point_grid(
            scan_points,
            family,
            args,
            progress_log=progress_log,
            stage_label="coarse_scan",
        )
        coarse_results = score_scan_results_near_gamma(
            coarse_results,
            family,
            args,
            path_kmax=0.03,
            points_per_side=max(3, args.near_gamma_points_per_side),
            progress_log=progress_log,
            stage_label="coarse_near_gamma",
        )

        top_candidates = select_top_scan_candidates(
            coarse_results,
            max_candidates=args.max_refine_centers,
            args=args,
        )
        selected_a_values = a_values
        selected_window_results = coarse_results
        if top_candidates:
            report_progress(
                f"Coarse scan window {round_idx} found {len(top_candidates)} candidate(s); stop expanding a",
                progress_log,
            )
            candidate_pool = top_candidates
            break
        report_progress(
            f"Coarse scan window {round_idx} found no valid candidates; expand a and continue",
            progress_log,
        )

    elapsed = time.time() - coarse_t0
    return selected_a_values, r_values, d_values, selected_window_results, candidate_pool, elapsed


def refine_candidates_3d(
    candidate_pool: Sequence[ScanPointResult],
    a_values: np.ndarray,
    r_values: np.ndarray,
    d_values: np.ndarray,
    family: FamilyConfig,
    args: argparse.Namespace,
    progress_log: Optional[List[str]] = None,
) -> List[ScanPointResult]:
    # Refine the local neighborhood around coarse-scan candidates.
    if not candidate_pool or len(a_values) == 0:
        return []

    if int(args.refine_bo_calls_per_center) > 0:
        refined_results = run_local_refinement_bo(
            candidate_pool,
            a_values,
            r_values,
            d_values,
            family,
            args,
            progress_log,
        )
        if refined_results:
            return refined_results

    refined_points = build_refined_scan_points_3d(
        candidate_pool,
        a_values,
        r_values,
        d_values,
        args,
    )
    report_progress(
        f"Refined scan ready | {len(candidate_pool)} centers, {len(refined_points)} sampled points",
        progress_log,
    )
    refined_results = run_scan_point_grid(
        refined_points,
        family,
        args,
        progress_log=progress_log,
        stage_label="refined_scan",
    )
    return score_scan_results_near_gamma(
        refined_results,
        family,
        args,
        path_kmax=0.03,
        points_per_side=max(3, args.near_gamma_points_per_side),
        progress_log=progress_log,
        stage_label="refined_near_gamma",
    )


def verify_candidates(
    candidates: Sequence[ScanPointResult],
    family: FamilyConfig,
    args: argparse.Namespace,
    progress_log: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, float]], Dict[str, Dict[str, np.ndarray]], float]:
    # Validate candidates on the final +/-0.07 path and return the first pass.
    t0 = time.time()
    path_results: Dict[str, Dict[str, np.ndarray]] = {}
    example = None

    for idx, candidate in enumerate(candidates, start=1):
        report_progress(
            (
                f"Validation candidate {idx} | a = {candidate.a_nm:.3f} nm, "
                f"hole radius ratio r/a = {candidate.r_over_a:.4f}, "
                f"h = {candidate.h_nm:.3f} nm"
            ),
            progress_log,
        )
        ok, path_result = verify_candidate_with_path(candidate, family, args)
        if ok:
            example = {
                "family_key": family.key,
                "pol": family.pol_label,
                "a_nm": candidate.a_nm,
                "r_over_a": candidate.r_over_a,
                "h_nm": candidate.h_nm,
                "d_over_a": candidate.d_over_a,
                "band": candidate.best_band_index,
                "lambda_nm": candidate.best_lambda_nm,
                "freq_im": candidate.best_freq_im,
                "gamma_q": candidate.best_q,
                "purcell_factor_est": candidate.purcell_factor_est,
                "mode_volume_a3": candidate.mode_volume_a3,
                "mode_volume_m3": candidate.mode_volume_m3,
                "validated": True,
            }
            path_results[family.key] = path_result
            break

    if example is None:
        report_progress("Validation found no candidate passing the k<=0.07 path check", progress_log)
    return example, path_results, time.time() - t0


def build_fallback_example(
    scan_results: Sequence[ScanPointResult],
    family: FamilyConfig,
    args: argparse.Namespace,
    progress_log: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, float]], Dict[str, Dict[str, np.ndarray]]]:
    valid = [item for item in scan_results if item.best_band_index >= 0]
    if not valid:
        return None, {}

    target_lambda = 0.5 * (args.range1_nm + args.range2_nm)
    best = max(
        valid,
        key=lambda item: (
            item.bic_count == 1,
            total_score_from_result(item, args),
            item.near_gamma_score,
            continuous_score_from_result(item, args),
            -abs(item.best_lambda_nm - target_lambda),
        ),
    )
    if progress_log is not None:
        report_progress(
            (
                "No strictly validated candidate was found; using the best fallback example | "
                f"a = {best.a_nm:.3f} nm, hole radius ratio r/a = {best.r_over_a:.4f}, "
                f"thickness ratio d/a = {best.d_over_a:.4f}, h = {best.h_nm:.3f} nm"
            ),
            progress_log,
        )

    example = {
        "family_key": family.key,
        "pol": family.pol_label,
        "a_nm": best.a_nm,
        "r_over_a": best.r_over_a,
        "h_nm": best.h_nm,
        "d_over_a": best.d_over_a,
        "band": best.best_band_index,
        "lambda_nm": best.best_lambda_nm,
        "freq_im": best.best_freq_im,
        "gamma_q": best.best_q,
        "purcell_factor_est": best.purcell_factor_est,
        "mode_volume_a3": best.mode_volume_a3,
        "mode_volume_m3": best.mode_volume_m3,
        "validated": False,
    }
    return example, {family.key: run_example_path(example, family, best.best_band_index, args)}


def build_signed_gamma_path(
    kmax: float, points_per_side: int
) -> Tuple[np.ndarray, np.ndarray]:
    # Build the signed path: left is M-Gamma, right is Gamma-X.
    left_axis = np.linspace(-kmax, 0.0, points_per_side, endpoint=False)
    right_axis = np.linspace(0.0, kmax, points_per_side)

    left_k = np.vstack([left_axis / math.sqrt(2.0), left_axis / math.sqrt(2.0)])
    right_k = np.vstack([right_axis, np.zeros_like(right_axis)])

    axis = np.concatenate([left_axis, right_axis])
    kpoints = np.hstack([left_k, right_k])
    return axis, kpoints


# -----------------------------------------------------------------------------
# Export and plotting
# -----------------------------------------------------------------------------


def run_example_path(
    example: Dict[str, float],
    family: FamilyConfig,
    target_band: int,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    # Run the short path around Gamma for the selected example geometry.
    _, phc = make_phc(example["d_over_a"], args.n_slab, args.n_hole, example["r_over_a"])
    axis, kpoints = build_signed_gamma_path(args.path_kmax, args.path_points_per_side)
    gamma_index = args.path_points_per_side
    numeig = max(args.numeig_path, target_band + 4)

    # Use a higher gmax for the final visualization path.
    gme = legume.GuidedModeExp(phc, gmax=args.validate_gmax, truncate_g="abs")
    gme.run(
        kpoints=kpoints,
        gmode_inds=list(family.gmode_inds),
        numeig=numeig,
        compute_im=True,
        gradients="approx",
        verbose=False,
    )

    freqs = np.asarray(gme.freqs)
    freqs_im = np.asarray(gme.freqs_im)
    q_vals = freqs / (2.0 * np.abs(freqs_im) + 1e-16)
    # Convert frequencies to wavelengths with safe division for display.
    wavelengths = np.divide(
        example["a_nm"],
        freqs,
        out=np.full_like(freqs, np.nan, dtype=float),
        where=np.abs(freqs) > 1e-12,
    )

    return {
        "axis": axis,
        "gamma_index": gamma_index,
        "freqs": freqs,
        "freqs_im": freqs_im,
        "q": q_vals,
        "wavelengths": wavelengths,
        "gme": gme,
    }


def save_scan_results_csv(path: str, scan_results: Sequence[ScanPointResult]) -> None:
    # Export the per-point scan summary as CSV.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "a_nm",
                "r_over_a",
                "h_nm",
                "d_over_a",
                "window_mode_count",
                "bic_count",
                "best_band_index",
                "best_freq_a_over_lambda",
                "best_freq_im",
                "best_Q",
                "best_lambda_nm",
                "near_gamma_score",
                "purcell_factor_est",
                "mode_volume_a3",
                "mode_volume_m3",
            ]
        )
        for item in scan_results:
            writer.writerow(
                [
                    f"{item.a_nm:.6f}",
                    f"{item.r_over_a:.6f}",
                    f"{item.h_nm:.6f}",
                    f"{item.d_over_a:.6f}",
                    item.window_mode_count,
                    item.bic_count,
                    item.best_band_index,
                    f"{item.best_freq:.9f}" if np.isfinite(item.best_freq) else "",
                    f"{item.best_freq_im:.6e}" if np.isfinite(item.best_freq_im) else "",
                    f"{item.best_q:.6e}" if np.isfinite(item.best_q) else "",
                    f"{item.best_lambda_nm:.6f}" if np.isfinite(item.best_lambda_nm) else "",
                    f"{item.near_gamma_score:.6f}" if np.isfinite(item.near_gamma_score) else "",
                    f"{item.purcell_factor_est:.6e}" if np.isfinite(item.purcell_factor_est) else "",
                    f"{item.mode_volume_a3:.6e}" if np.isfinite(item.mode_volume_a3) else "",
                    f"{item.mode_volume_m3:.6e}" if np.isfinite(item.mode_volume_m3) else "",
                ]
            )


def save_robustness_csv(path: str, summaries: Sequence[RobustCandidateSummary]) -> None:
    # Export the post-check fabrication-tolerance statistics.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "a_nm",
                "r_over_a",
                "h_nm",
                "d_over_a",
                "nominal_score",
                "mean_score",
                "std_score",
                "robust_score",
                "score_q10",
                "yield_fraction",
                "valid_fraction",
                "passing_samples",
                "valid_samples",
                "total_samples",
            ]
        )
        for item in summaries:
            writer.writerow(
                [
                    item.rank,
                    f"{item.a_nm:.6f}",
                    f"{item.r_over_a:.6f}",
                    f"{item.h_nm:.6f}",
                    f"{item.d_over_a:.6f}",
                    f"{item.nominal_score:.6f}" if np.isfinite(item.nominal_score) else "",
                    f"{item.mean_score:.6f}" if np.isfinite(item.mean_score) else "",
                    f"{item.std_score:.6f}" if np.isfinite(item.std_score) else "",
                    f"{item.robust_score:.6f}" if np.isfinite(item.robust_score) else "",
                    f"{item.score_q10:.6f}" if np.isfinite(item.score_q10) else "",
                    f"{item.yield_fraction:.6f}",
                    f"{item.valid_fraction:.6f}",
                    item.passing_samples,
                    item.valid_samples,
                    item.total_samples,
                ]
            )


def plot_phase_diagrams_r_slices(
    outpath: str,
    phase_points_by_r: Dict[float, List[ScanPointResult]],
    example: Optional[Dict[str, float]],
    family_lookup: Dict[str, FamilyConfig],
    args: argparse.Namespace,
) -> plt.Figure:
    # Show the scan result as phase maps over the selected hole radius ratio slices.
    family = family_lookup[next(iter(family_lookup))]
    r_keys = list(phase_points_by_r.keys())
    ncols = len(r_keys)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    phase_cmap = ListedColormap(["#2166ac", "#7b1fa2", "#c62828"])
    phase_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], phase_cmap.N)

    all_points = [item for points in phase_points_by_r.values() for item in points]
    if all_points:
        all_a = np.array([item.a_nm for item in all_points], dtype=float)
        all_d = np.array([item.d_over_a for item in all_points], dtype=float)
        a_span = max(float(np.max(all_a) - np.min(all_a)), 1.0)
        d_span = max(float(np.max(all_d) - np.min(all_d)), 1e-3)
        xlim = (float(np.min(all_a) - 0.03 * a_span), float(np.max(all_a) + 0.03 * a_span))
        ylim = (float(np.min(all_d) - 0.05 * d_span), float(np.max(all_d) + 0.05 * d_span))
    else:
        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)
    im = None
    for ax, r_value in zip(axes, r_keys):
        points = phase_points_by_r[r_value]
        if points:
            a_plot = np.array([item.a_nm for item in points], dtype=float)
            d_plot = np.array([item.d_over_a for item in points], dtype=float)
            phase_level = np.array(
                [
                    2.0
                    if (item.window_mode_count == 1 and item.bic_count == 1)
                    else 1.0
                    if item.bic_count == 1
                    else 0.0
                    for item in points
                ],
                dtype=float,
            )
            im = ax.scatter(
                a_plot,
                d_plot,
                c=phase_level,
                cmap=phase_cmap,
                norm=phase_norm,
                s=180,
                marker="s",
                edgecolors="none",
            )
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(
            f"{family.title} | hole radius ratio r/a = {r_value:.3f}",
            pad=DEFAULT_AX_TITLE_PAD,
        )
        ax.set_xlabel("period a (nm)")
        ax.set_ylabel("thickness ratio d/a")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if example is not None and abs(example["r_over_a"] - r_value) < 1e-9:
            ax.scatter(
                [example["a_nm"]],
                [example["d_over_a"]],
                s=90,
                marker="*",
                edgecolors="white",
                facecolors="none",
                linewidths=1.2,
            )

    legend_handles = [
        Patch(facecolor="#c62828", edgecolor="none", label="Red: one target-window mode and one Gamma BIC"),
        Patch(facecolor="#7b1fa2", edgecolor="none", label="Purple: Gamma BIC exists but target-window mode count is not exactly one"),
        Patch(facecolor="#2166ac", edgecolor="none", label="Blue: no Gamma BIC in the target window"),
    ]
    if example is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=9,
                label="Selected example",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="phase meaning",
        frameon=False,
    )

    fig.suptitle(
        (
            f"{args.project_name} | phase map in a vs thickness ratio d/a "
            f"with scanned hole radius ratio r/a slices, target window "
            f"{args.range1_nm:.1f}-{args.range2_nm:.1f} nm"
        ),
        y=DEFAULT_SUPTITLE_Y,
        fontsize=17,
    )
    fig.savefig(outpath, dpi=220)
    return fig


def plot_bo_pred_vs_true(outpath: str, bo_eval: Dict[str, np.ndarray], args: argparse.Namespace) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    pred = bo_eval.get("pred_score", np.array([], dtype=float))
    true = bo_eval.get("true_score", np.array([], dtype=float))
    mask = np.isfinite(pred) & np.isfinite(true)

    if np.any(mask):
        ax.scatter(true[mask], pred[mask], s=42, c="#1565c0", alpha=0.8, edgecolors="none", label="BO samples")
        lo = float(min(np.min(true[mask]), np.min(pred[mask])))
        hi = float(max(np.max(true[mask]), np.max(pred[mask])))
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="black", label="Ideal parity")
    else:
        ax.text(0.5, 0.5, "No valid BO predictions yet", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("BO prediction vs true score", pad=DEFAULT_AX_TITLE_PAD, fontsize=15)
    ax.set_xlabel("true score", fontsize=17)
    ax.set_ylabel("predicted score", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(alpha=0.25)
    if np.any(mask):
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=15)
    fig.suptitle(f"{args.project_name} | BO parity plot", y=DEFAULT_SUPTITLE_Y, fontsize=16)
    fig.savefig(outpath, dpi=220)
    return fig


def plot_bo_error_vs_iter(outpath: str, bo_eval: Dict[str, np.ndarray], args: argparse.Namespace) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    iteration = bo_eval.get("iteration", np.array([], dtype=float))
    abs_error = bo_eval.get("abs_error", np.array([], dtype=float))
    mask = np.isfinite(iteration) & np.isfinite(abs_error)

    if np.any(mask):
        ax.plot(iteration[mask], abs_error[mask], marker="o", ms=4, lw=1.2, color="#c62828", label="Absolute error")
        if np.count_nonzero(mask) >= 2:
            window = min(5, np.count_nonzero(mask))
            kernel = np.ones(window) / window
            smooth = np.convolve(abs_error[mask], kernel, mode="same")
            ax.plot(iteration[mask], smooth, lw=1.6, color="#ef6c00", alpha=0.9, label="Smoothed trend")
    else:
        ax.text(0.5, 0.5, "No BO error data yet", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("BO absolute prediction error", pad=DEFAULT_AX_TITLE_PAD, fontsize=15)
    ax.set_xlabel("BO iteration", fontsize=17)
    ax.set_ylabel("absolute prediction error", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(alpha=0.25)
    if np.any(mask):
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=15)
    fig.suptitle(f"{args.project_name} | BO error vs iteration", y=DEFAULT_SUPTITLE_Y, fontsize=16)
    fig.savefig(outpath, dpi=220)
    return fig


def plot_bo_ci_coverage(outpath: str, bo_eval: Dict[str, np.ndarray], args: argparse.Namespace) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    iteration = bo_eval.get("iteration", np.array([], dtype=float))
    covered = bo_eval.get("covered_95", np.array([], dtype=float))
    mask = np.isfinite(iteration) & np.isfinite(covered)

    if np.any(mask):
        cumulative = np.cumsum(covered[mask]) / np.arange(1, np.count_nonzero(mask) + 1)
        ax.plot(iteration[mask], cumulative, marker="o", ms=4, lw=1.4, color="#2e7d32", label="Cumulative coverage")
        ax.axhline(0.95, ls="--", lw=1.0, color="black", label="95% target")
        ax.set_ylim(0.0, 1.05)
    else:
        ax.text(0.5, 0.5, "No BO coverage data yet", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("95% confidence interval coverage", pad=DEFAULT_AX_TITLE_PAD, fontsize=15)
    ax.set_xlabel("BO iteration", fontsize=17)
    ax.set_ylabel("cumulative coverage", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(alpha=0.25)
    if np.any(mask):
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=15)
    fig.suptitle(f"{args.project_name} | BO CI coverage", y=DEFAULT_SUPTITLE_Y, fontsize=16)
    fig.savefig(outpath, dpi=220)
    return fig


def plot_candidate_purcell(
    outpath: str,
    candidates: Sequence[ScanPointResult],
    args: argparse.Namespace,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.2, 6.8), constrained_layout=True)
    ordered = list(candidates)

    if ordered:
        ranks = np.arange(1, len(ordered) + 1)
        valid_mask = np.array(
            [np.isfinite(item.purcell_factor_est) and item.purcell_factor_est > 0 for item in ordered],
            dtype=bool,
        )
        colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(ordered)))
        if np.any(valid_mask):
            purcell = np.array([item.purcell_factor_est for item in ordered], dtype=float)
            ax.scatter(
                ranks[valid_mask],
                purcell[valid_mask],
                s=180,
                c=colors[valid_mask],
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )
            ax.set_yscale("log")
            for rank, value in zip(ranks[valid_mask], purcell[valid_mask]):
                ax.text(rank, value * 1.10, f"{value:.2e}", ha="center", va="bottom")
        if np.any(~valid_mask):
            ax.scatter(
                ranks[~valid_mask],
                np.full(np.count_nonzero(~valid_mask), 1.0),
                s=180,
                marker="x",
                c="#9e9e9e",
                linewidths=2.0,
                zorder=3,
            )
            for rank in ranks[~valid_mask]:
                ax.text(rank, 1.12, "N/A", ha="center", va="bottom", color="#616161")

        ax.set_xticks(
            ranks,
            [
                f"Candidate {idx}\na={item.a_nm:.1f} nm\nr/a={item.r_over_a:.3f}\nd/a={item.d_over_a:.3f}"
                for idx, item in zip(ranks, ordered)
            ],
        )
        ax.tick_params(axis="x", pad=12)
        ax.set_xlim(0.5, len(ordered) + 0.5)
    else:
        ax.text(0.5, 0.5, "No candidate Purcell estimates yet", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Shortlisted candidates: approximate Purcell factor", pad=DEFAULT_AX_TITLE_PAD)
    ax.set_xlabel("candidate index and geometry parameters")
    ax.set_ylabel("approximate unit-cell Purcell factor")
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"{args.project_name} | candidate Purcell estimates", y=DEFAULT_SUPTITLE_Y)
    fig.savefig(outpath, dpi=220)
    return fig


def plot_candidate_score_breakdown(
    outpath: str,
    candidates: Sequence[ScanPointResult],
    args: argparse.Namespace,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13.8, 7.6), constrained_layout=True)
    ordered = list(candidates)

    if ordered:
        ranks = np.arange(1, len(ordered) + 1, dtype=float)
        labels = [
            f"Candidate {idx}\na={item.a_nm:.1f} nm\nr/a={item.r_over_a:.3f}\nd/a={item.d_over_a:.3f}"
            for idx, item in zip(ranks.astype(int), ordered)
        ]
        term_specs = [
            ("loss_term", "loss", "#1565c0"),
            ("q_term", "logQ", "#2e7d32"),
            ("thickness_term", "thickness", "#6a1b9a"),
            ("single_mode_term", "single mode", "#ef6c00"),
            ("single_bic_term", "single BIC", "#ad1457"),
            ("lambda_penalty", "-lambda penalty", "#757575"),
            ("near_gamma_term", "near-Gamma", "#00838f"),
        ]
        width = 0.10
        offsets = (np.arange(len(term_specs), dtype=float) - 0.5 * (len(term_specs) - 1)) * width
        total_scores: List[float] = []

        for offset, (key, label, color) in zip(offsets, term_specs):
            values: List[float] = []
            for item in ordered:
                terms = score_terms_from_result(item, args)
                if key == "lambda_penalty":
                    values.append(-float(terms["lambda_penalty"]))
                elif key == "near_gamma_term":
                    values.append(max(float(item.near_gamma_score), 0.0))
                else:
                    values.append(float(terms[key]))
                total_scores.append(total_score_from_result(item, args))
            ax.bar(ranks + offset, values, width=width, color=color, alpha=0.90, label=label)

        ax.plot(
            ranks,
            [total_score_from_result(item, args) for item in ordered],
            color="black",
            lw=2.2,
            marker="o",
            ms=7,
            label="total score",
            zorder=5,
        )
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.45)
        ax.set_xticks(ranks, labels)
        ax.tick_params(axis="x", pad=12)
        ax.set_xlim(0.35, len(ordered) + 0.65)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    else:
        ax.text(0.5, 0.5, "No candidate objective data yet", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Candidate objective breakdown", pad=DEFAULT_AX_TITLE_PAD)
    ax.set_xlabel("candidate index and geometry parameters")
    ax.set_ylabel("score contribution")
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"{args.project_name} | candidate objective terms", y=DEFAULT_SUPTITLE_Y)
    fig.savefig(outpath, dpi=220)
    return fig


def plot_robustness_summary(
    outpath: str,
    summaries: Sequence[RobustCandidateSummary],
    args: argparse.Namespace,
) -> plt.Figure:
    fig, ax_left = plt.subplots(figsize=(12.6, 7.2), constrained_layout=True)
    ax_right = ax_left.twinx()
    ordered = list(summaries)

    if ordered:
        ranks = np.arange(1, len(ordered) + 1, dtype=float)
        labels = [
            f"Candidate {item.rank}\na={item.a_nm:.1f} nm\nr/a={item.r_over_a:.3f}\nd/a={item.d_over_a:.3f}"
            for item in ordered
        ]
        nominal = np.array([item.nominal_score for item in ordered], dtype=float)
        robust = np.array([item.robust_score for item in ordered], dtype=float)
        q10 = np.array([item.score_q10 for item in ordered], dtype=float)
        yield_frac = np.array([item.yield_fraction for item in ordered], dtype=float)
        valid_frac = np.array([item.valid_fraction for item in ordered], dtype=float)

        ax_right.bar(ranks, yield_frac, width=0.46, color="#ffccbc", alpha=0.80, label="yield")
        ax_right.plot(
            ranks,
            valid_frac,
            color="#6d4c41",
            lw=1.8,
            marker="s",
            ms=6,
            label="valid fraction",
        )
        ax_left.plot(ranks, nominal, color="#1565c0", lw=2.0, marker="o", ms=7, label="nominal score")
        ax_left.plot(ranks, robust, color="#c62828", lw=2.0, marker="D", ms=7, label="robust score")
        ax_left.plot(ranks, q10, color="#2e7d32", lw=1.8, marker="^", ms=7, label="q10 score")

        for rank, yv in zip(ranks, yield_frac):
            ax_right.text(rank, min(1.02, yv + 0.04), f"{yv:.2f}", ha="center", va="bottom")

        ax_left.set_xticks(ranks, labels)
        ax_left.tick_params(axis="x", pad=12)
        ax_left.set_xlim(0.5, len(ordered) + 0.5)
        ax_right.set_ylim(0.0, 1.08)

        handles_left, labels_left = ax_left.get_legend_handles_labels()
        handles_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(
            handles_left + handles_right,
            labels_left + labels_right,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )
    else:
        ax_left.text(0.5, 0.5, "No robustness data yet", ha="center", va="center", transform=ax_left.transAxes)

    ax_left.set_title("Fabrication robustness summary", pad=DEFAULT_AX_TITLE_PAD)
    ax_left.set_xlabel("candidate index and geometry parameters")
    ax_left.set_ylabel("score")
    ax_right.set_ylabel("yield / valid fraction")
    ax_left.grid(alpha=0.25, axis="y")
    fig.suptitle(f"{args.project_name} | robustness screen", y=DEFAULT_SUPTITLE_Y)
    fig.savefig(outpath, dpi=220)
    return fig


def plot_example_dispersion(
    outpath: str,
    example: Dict[str, float],
    path_results: Dict[str, Dict[str, np.ndarray]],
    family_lookup: Dict[str, FamilyConfig],
    args: argparse.Namespace,
) -> plt.Figure:
    # Left: near-Gamma band scatter colored by Q. Right: unit-cell near-field |E|.
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 6.8), constrained_layout=True)
    display_min = args.range1_nm - args.display_margin_nm
    display_max = args.range2_nm + args.display_margin_nm
    freq_min = example["a_nm"] / display_max
    freq_max = example["a_nm"] / display_min

    def freq_to_lambda(freq: np.ndarray) -> np.ndarray:
        freq_arr = np.asarray(freq, dtype=float)
        return np.divide(
            example["a_nm"],
            freq_arr,
            out=np.full_like(freq_arr, np.nan, dtype=float),
            where=np.abs(freq_arr) > 1e-12,
        )

    def lambda_to_freq(wavelength: np.ndarray) -> np.ndarray:
        lam_arr = np.asarray(wavelength, dtype=float)
        return np.divide(
            example["a_nm"],
            lam_arr,
            out=np.full_like(lam_arr, np.nan, dtype=float),
            where=np.abs(lam_arr) > 1e-12,
        )

    family = family_lookup[next(iter(family_lookup))]
    result = path_results[family.key]
    axis = result["axis"]
    wavelengths = result["wavelengths"]
    freqs = result["freqs"]
    q_vals = result["q"]
    gamma_index = int(result["gamma_index"])
    target_band = int(example["band"])

    bands_to_plot: List[int] = []
    for band in range(freqs.shape[1]):
        band_lam = wavelengths[:, band]
        in_window = np.any(
            np.isfinite(band_lam)
            & (band_lam >= display_min)
            & (band_lam <= display_max)
        )
        if in_window or band == target_band:
            bands_to_plot.append(band)
    if target_band not in bands_to_plot:
        bands_to_plot.append(target_band)
    bands_to_plot = sorted(set(bands_to_plot))

    log_q_values: List[np.ndarray] = []
    for band in bands_to_plot:
        band_mask = (
            np.isfinite(wavelengths[:, band])
            & (wavelengths[:, band] >= display_min)
            & (wavelengths[:, band] <= display_max)
        ) | (band == target_band)
        if np.any(band_mask):
            log_q_values.append(
                np.log10(
                    np.clip(
                        q_vals[band_mask, band],
                        1.0,
                        None,
                    )
                )
            )
    if log_q_values:
        log_q_all = np.concatenate(log_q_values)
        log_q_min = float(np.nanmin(log_q_all))
        log_q_max = float(np.nanmax(log_q_all))
        if not np.isfinite(log_q_min) or not np.isfinite(log_q_max):
            log_q_min, log_q_max = 0.0, 1.0
        elif abs(log_q_max - log_q_min) < 1e-12:
            log_q_max = log_q_min + 1.0
    else:
        log_q_min, log_q_max = 0.0, 1.0

    ax_band = axes[0]
    scatter_artist = None
    other_band_handle = None
    target_band_handle = None
    window_handle = None
    gamma_line_handle = None
    for band in bands_to_plot:
        band_mask = (
            np.isfinite(wavelengths[:, band])
            & (wavelengths[:, band] >= display_min)
            & (wavelengths[:, band] <= display_max)
        ) | (band == target_band)
        point_size = 26 if band == target_band else 14
        scatter_artist = ax_band.scatter(
            axis[band_mask],
            freqs[band_mask, band],
            c=np.log10(np.clip(q_vals[band_mask, band], 1.0, None)),
            cmap="viridis",
            vmin=log_q_min,
            vmax=log_q_max,
            s=point_size,
            edgecolors="none",
            alpha=0.9 if band == target_band else 0.65,
        )
        if band == target_band and target_band_handle is None:
            target_band_handle = Line2D([0], [0], marker="o", linestyle="None", color="#238b45", markersize=6, label="Target band")
        elif band != target_band and other_band_handle is None:
            other_band_handle = Line2D([0], [0], marker="o", linestyle="None", color="#5e81ac", markersize=5, label="Other bands")
    ax_band.axvline(0.0, color="black", lw=0.8, ls="--")
    ax_band.axhspan(example["a_nm"] / args.range2_nm, example["a_nm"] / args.range1_nm, color="#ffccbc", alpha=0.35)
    gamma_line_handle = Line2D([0], [0], color="black", lw=0.8, ls="--", label="Gamma point")
    window_handle = Patch(facecolor="#ffccbc", edgecolor="none", alpha=0.35, label="Target wavelength window")
    ax_band.set_title(
        f"{family.title}: near-Gamma band scatter",
        pad=DEFAULT_AX_TITLE_PAD,
        fontsize=15,
    )
    ax_band.set_ylabel("normalized frequency a/λ", fontsize=20)
    ax_band.set_xlabel("k∥ a / (2π)", fontsize=20)
    ax_band.set_xlim(axis[0], axis[-1])
    freq_padding = 0.10 * max(freq_max - freq_min, 1e-6)
    ax_band.set_ylim(freq_min, freq_max + freq_padding)
    ax_band.grid(alpha=0.25)
    ax_band.tick_params(axis="both", labelsize=17)
    ax_band.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: "0" if abs(float(x)) < 1e-12 else f"{float(x):g}")
    )
    secax = ax_band.secondary_yaxis("right", functions=(freq_to_lambda, lambda_to_freq))
    secax.set_ylabel("wavelength (nm)", fontsize=18)
    secax.set_yticks([500, 520, 540, 560])
    secax.tick_params(axis="y", labelsize=16)
    if scatter_artist is not None:
        cbar = fig.colorbar(scatter_artist, ax=ax_band, shrink=0.9)
        cbar.set_label("log10(Q)", fontsize=18)
        cbar.ax.tick_params(labelsize=16)

    ax_band.text(
        0.5,
        0.995,
        "M-Γ-X",
        transform=ax_band.transAxes,
        ha="center",
        va="top",
        fontsize=21,
        fontfamily="Arial",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        zorder=6,
    )

    # Plot the near field on the slab mid-plane inside one unit cell.
    ax_field = axes[1]
    gme = result["gme"]
    field_dict, xgrid, ygrid = gme.get_field_xy(
        "E",
        kind=gamma_index,
        mind=target_band,
        z=example["d_over_a"] / 2.0,
        component="xyz",
        Nx=181,
        Ny=181,
    )
    ex = field_dict["x"]
    ey = field_dict["y"]
    ez = field_dict["z"]
    e_abs = np.real(np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2))
    e_abs_max = float(np.nanmax(e_abs)) if np.size(e_abs) > 0 else 0.0
    if e_abs_max > 0:
        e_abs = e_abs / e_abs_max
    eps_xy, _, _ = gme.get_eps_xy(z=example["d_over_a"] / 2.0, xgrid=xgrid, ygrid=ygrid)
    eps_xy = np.real(eps_xy)
    img = ax_field.imshow(
        e_abs.T,
        origin="lower",
        extent=[xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]],
        cmap="inferno",
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )
    ax_field.contour(
        xgrid,
        ygrid,
        eps_xy.T,
        levels=[0.5 * (args.n_slab**2 + args.n_hole**2)],
        colors=["white"],
        linewidths=0.8,
    )
    ax_field.set_title("Periodic-cell near-field |E| at Gamma", pad=DEFAULT_AX_TITLE_PAD, fontsize=15)
    ax_field.set_xlabel("x / a", fontsize=20)
    ax_field.set_ylabel("y / a", fontsize=20)
    ax_field.tick_params(axis="both", labelsize=17)
    cbar_field = fig.colorbar(img, ax=ax_field, shrink=0.9)
    cbar_field.set_label("normalized |E|", fontsize=18)
    cbar_field.ax.tick_params(labelsize=16)

    legend_handles = [handle for handle in (target_band_handle, other_band_handle, gamma_line_handle, window_handle) if handle is not None]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
        )

    fig.suptitle(
        f"{args.project_name} | Example geometry: a = {example['a_nm']:.3f} nm, thickness ratio d/a = {example['d_over_a']:.4f}, "
        f"near-Gamma path = +/-{args.path_kmax:.3f}, display window = {display_min:.1f}-{display_max:.1f} nm",
        y=DEFAULT_SUPTITLE_Y,
        fontsize=17,
    )
    fig.savefig(outpath, dpi=220)
    return fig


def save_summary(
    outpath: str,
    args: argparse.Namespace,
    acceleration: Dict[str, bool],
    a_min_nm: float,
    a_max_nm: float,
    scan_results: Sequence[ScanPointResult],
    example: Optional[Dict[str, float]],
    stage_times: Dict[str, float],
    progress_log: Sequence[str],
    bo_eval: Optional[Dict[str, np.ndarray]] = None,
    robust_summaries: Optional[Sequence[RobustCandidateSummary]] = None,
    robust_best: Optional[RobustCandidateSummary] = None,
) -> None:
    # Write the run configuration, timing, and selected example into a text summary.
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(f"Project: {args.project_name}\n")
        f.write("Fast Gamma-point BIC search summary (V4)\n")
        f.write("=" * 40 + "\n")
        f.write(f"target wavelength window: {args.range1_nm:.3f} - {args.range2_nm:.3f} nm\n")
        f.write(describe_scan_mode(args) + "\n")
        f.write(f"n_slab = {args.n_slab:.6f}\n")
        f.write(f"n_hole = {args.n_hole:.6f}\n")
        f.write(f"hole radius ratio r/a scan = {format_r_scan_description(args)}\n")
        f.write(f"thickness ratio d/a scan = [{args.d_min:.4f}, {args.d_max:.4f}], {args.d_points} points\n")
        f.write(f"minimum physical h kept = {args.min_h_nm:.4f} nm\n")
        f.write(f"period window used for plots/listing = [{a_min_nm:.4f}, {a_max_nm:.4f}] nm, {args.a_points} points per round\n")
        f.write(f"fast test mode = {bool(getattr(args, 'fast_test', False))}\n")
        f.write(f"coarse bootstrap points per window = {args.coarse_bootstrap_points}\n")
        f.write(f"parallel workers = {getattr(args, 'parallel_workers', 1)}\n")
        f.write(f"BO calls = {args.bo_calls}\n")
        f.write(f"threads per worker = {args.threads_per_worker}\n")
        f.write(f"BO rounds = {args.bo_rounds}\n")
        f.write(f"BO random starts = {args.bo_random_starts}\n")
        f.write(f"refine BO calls per center = {args.refine_bo_calls_per_center}\n")
        f.write(f"refine BO random starts = {args.refine_bo_random_starts}\n")
        f.write(f"Purcell shortlist evals per stage = {args.purcell_top_evals}\n")
        f.write(f"Purcell rank weight = {args.purcell_weight:.4f}\n")
        f.write(f"Purcell field grid = {args.purcell_xy_samples} x {args.purcell_xy_samples} x {args.purcell_z_samples}\n")
        f.write(f"fabrication robustness candidates = {args.robust_top_candidates}\n")
        f.write(f"fabrication robustness samples/candidate = {args.robust_samples}\n")
        f.write(
            "fabrication sigma (nm) = "
            f"(a {args.robust_a_sigma_nm:.3f}, radius {args.robust_radius_sigma_nm:.3f}, h {args.robust_h_sigma_nm:.3f})\n"
        )
        f.write(f"fabrication robustness beta = {args.robust_beta:.4f}\n")
        f.write(f"subwavelength only = {not args.allow_superwavelength}\n")
        f.write("primary screening metric = continuous score with loss proxy, log10(Q), wavelength penalty, and mode-count penalties\n")
        f.write("Purcell is report-only and is excluded from BO and candidate ranking\n")
        f.write(
            "Purcell uses a slab-confined unit-cell estimate: "
            "Fp = (3/4pi^2) * (lambda_m/n_eff)^3 * Q / Veff_m^3\n"
        )
        f.write(
            "n_eff is taken from Legume's internal effective-layer permittivity; "
            "Veff integrates only over one patterned slab cell and is converted to m^3.\n"
        )
        f.write(
            "fabrication robustness uses Gamma-only Monte Carlo perturbations in SI-labeled nm dimensions "
            "and reports mean(score), std(score), q10(score), and yield.\n"
        )
        f.write(f"GPU-related packages detected: {acceleration}\n")
        f.write("\nTiming summary\n")
        f.write(f"coarse scan elapsed = {stage_times.get('coarse_scan', 0.0):.2f} s\n")
        f.write(f"Bayes opt elapsed = {stage_times.get('bayes_opt', 0.0):.2f} s\n")
        f.write(f"refinement elapsed = {stage_times.get('refinement', 0.0):.2f} s\n")
        f.write(f"validation elapsed = {stage_times.get('verification', 0.0):.2f} s\n")
        f.write(f"robustness elapsed = {stage_times.get('robustness', 0.0):.2f} s\n")
        f.write(f"total elapsed = {stage_times.get('total', 0.0):.2f} s\n")
        f.write(f"selected polarization family = {args.pol.upper()}\n")
        f.write(f"scan points recorded = {len(scan_results)}\n")
        f.write(
            f"points with exactly one mode in target window and one Gamma BIC = "
            f"{sum((item.window_mode_count == 1 and item.bic_count == 1) for item in scan_results)}\n"
        )
        f.write(f"points with exactly one mode in target window = {sum(item.window_mode_count == 1 for item in scan_results)}\n")
        f.write(f"points with exactly one BIC = {sum(item.bic_count == 1 for item in scan_results)}\n")
        if bo_eval is not None and bo_eval.get("iteration", np.array([])).size > 0:
            abs_error = bo_eval.get("abs_error", np.array([], dtype=float))
            covered = bo_eval.get("covered_95", np.array([], dtype=float))
            finite_error = abs_error[np.isfinite(abs_error)]
            finite_cov = covered[np.isfinite(covered)]
            f.write("\nBO surrogate diagnostics\n")
            if finite_error.size > 0:
                f.write(f"BO MAE = {np.mean(finite_error):.6f}\n")
            if finite_cov.size > 0:
                f.write(f"BO 95% CI coverage = {np.mean(finite_cov):.4f}\n")
        if args.bo_calls > 0:
            f.write("\nBO analysis figures\n")
            f.write("bo_pred_vs_true.png\n")
            f.write("bo_error_vs_iter.png\n")
            f.write("bo_ci_coverage.png\n")
        f.write("\nCandidate analysis figures\n")
        f.write("candidate_purcell.png\n")
        if robust_summaries:
            f.write("\nFabrication robustness outputs\n")
            f.write("robustness_candidates.csv\n")
            if robust_best is not None:
                f.write(
                    "robust_best = "
                    f"rank {robust_best.rank}, a {robust_best.a_nm:.6f} nm, "
                    f"r/a {robust_best.r_over_a:.6f}, d/a {robust_best.d_over_a:.6f}, "
                    f"yield {robust_best.yield_fraction:.4f}, robust_score {robust_best.robust_score:.6f}, "
                    f"q10 {robust_best.score_q10:.6f}\n"
                )
        if example is not None:
            f.write("\nSelected example geometry\n")
            f.write(f"validated_example = {bool(example.get('validated', False))}\n")
            f.write(f"polarization = {example['pol']}\n")
            f.write(f"a = {example['a_nm']:.6f} nm\n")
            f.write(f"hole radius ratio r/a = {example['r_over_a']:.6f}\n")
            f.write(f"h = {example['h_nm']:.6f} nm\n")
            f.write(f"thickness ratio d/a = {example['d_over_a']:.6f}\n")
            f.write(f"target Gamma wavelength = {example['lambda_nm']:.6f} nm\n")
            f.write(f"target band index = {int(example['band'])}\n")
            f.write(f"target Gamma freqs_im = {example['freq_im']:.6e}\n")
            f.write(f"target Gamma Q = {example['gamma_q']:.6e}\n")
            if np.isfinite(example.get("purcell_factor_est", math.nan)):
                f.write(f"approximate Purcell factor = {example['purcell_factor_est']:.6e}\n")
            if np.isfinite(example.get("mode_volume_a3", math.nan)):
                f.write(f"approximate mode volume (a^3) = {example['mode_volume_a3']:.6e}\n")
            if np.isfinite(example.get("mode_volume_m3", math.nan)):
                f.write(f"approximate mode volume (m^3) = {example['mode_volume_m3']:.6e}\n")
        else:
            f.write("\nNo example geometry with exact-one intervals was found for the selected polarization.\n")

        if robust_summaries:
            f.write("\nFabrication robustness ranking\n")
            for item in robust_summaries:
                f.write(
                    f"rank {item.rank}: a = {item.a_nm:.6f} nm, r/a = {item.r_over_a:.6f}, "
                    f"d/a = {item.d_over_a:.6f}, yield = {item.yield_fraction:.4f}, "
                    f"valid = {item.valid_fraction:.4f}, robust_score = {item.robust_score:.6f}, "
                    f"mean = {item.mean_score:.6f}, std = {item.std_score:.6f}, q10 = {item.score_q10:.6f}\n"
                )

        if progress_log:
            f.write("\nProgress log\n")
            f.write("-" * 40 + "\n")
            for line in progress_log:
                f.write(line + "\n")


# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------


def _old_main() -> None:
    # Legacy alias kept for backward compatibility.
    main_run()


def _main_prev() -> None:
    # Legacy alias kept for backward compatibility.
    main_run()


def main() -> None:
    # Public entry point alias.
    main_run()


def main_run() -> None:
    # Active script entry point used by `python findBIC.py`.
    args = parse_args()
    if args.fast_test:
        fast_changes = apply_fast_test_profile(args)
        if fast_changes:
            print("Fast test profile enabled: " + ", ".join(fast_changes))
    if args.range2_nm <= args.range1_nm:
        raise ValueError("range2_nm must be larger than range1_nm")
    r_values_check = build_r_values_over_a(args)
    if len(r_values_check) < 1:
        raise ValueError("at least one r/a value is required")
    if args.r_points < 1:
        raise ValueError("r_points >= 1 is required")
    if args.coarse_bootstrap_points < 1:
        raise ValueError("coarse_bootstrap_points >= 1 is required")
    if args.bo_rounds < 1:
        raise ValueError("bo_rounds >= 1 is required")
    if args.scan_mode != "r_d" and args.a_points < 2:
        raise ValueError("a_points >= 2 is required when a is scanned")
    if args.scan_mode in ("a_r_d", "a_d", "r_d") and args.d_points < 2:
        raise ValueError("d_points >= 2 is required when d/a is scanned")
    if args.refine_bo_calls_per_center < 0:
        raise ValueError("refine_bo_calls_per_center >= 0 is required")
    if args.refine_bo_random_starts < 0:
        raise ValueError("refine_bo_random_starts >= 0 is required")
    if args.purcell_top_evals < 0:
        raise ValueError("purcell_top_evals >= 0 is required")
    if args.purcell_xy_samples < 5:
        raise ValueError("purcell_xy_samples >= 5 is required")
    if args.purcell_z_samples < 3:
        raise ValueError("purcell_z_samples >= 3 is required")
    if args.threads_per_worker < 1:
        raise ValueError("threads_per_worker >= 1 is required")
    if args.parallel_chunk_size < 0:
        raise ValueError("parallel_chunk_size >= 0 is required")

    args.parallel_workers = resolve_parallel_workers(args)
    if args.parallel_workers > 1:
        configure_parallel_env(args.threads_per_worker)

    configure_matplotlib(show_plots=not args.no_show_plots)

    base_outdir = os.path.abspath(args.outdir)
    outdir = make_timestamped_output_dir(base_outdir, args.project_name)
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    print(f"Project: {args.project_name}")
    acceleration = detect_acceleration()
    active_family = get_active_family(args.pol)
    progress_log: List[str] = []
    stage_times: Dict[str, float] = {}
    args.min_h_nm = max(args.min_h_nm, 60.0)
    report_progress(
        (
            f"Parallel runtime | workers = {args.parallel_workers}, "
            f"threads/worker = {args.threads_per_worker}"
        ),
        progress_log,
    )
    report_progress(describe_scan_mode(args), progress_log)
    report_progress(
        (
            f"BO-driven search | coarse bootstrap <= {args.coarse_bootstrap_points} points/window, "
            f"BO calls = {args.bo_calls}, BO rounds = {args.bo_rounds}, "
            f"random starts = {args.bo_random_starts}"
        ),
        progress_log,
    )
    report_progress(
        (
            f"Refinement strategy | local BO calls/center = {args.refine_bo_calls_per_center}, "
            f"random starts = {args.refine_bo_random_starts}, robustness shortlist = {args.robust_top_candidates}"
        ),
        progress_log,
    )

    selected_a_values, r_values, d_values, selected_window_results, candidate_pool, coarse_elapsed = (
        coarse_scan_with_adaptive_a(active_family, args, progress_log)
    )
    record_stage_time(stage_times, "coarse_scan", "coarse scan", coarse_elapsed, progress_log)

    combined_results = list(selected_window_results)
    bo_eval = {
        "iteration": np.array([], dtype=float),
        "pred_score": np.array([], dtype=float),
        "pred_std": np.array([], dtype=float),
        "true_score": np.array([], dtype=float),
        "abs_error": np.array([], dtype=float),
        "covered_95": np.array([], dtype=float),
    }
    bo_available = Optimizer is not None and Real is not None and Categorical is not None
    if len(selected_a_values) > 0 and args.bo_calls > 0 and bo_available:
        report_progress(
            f"Bayes opt ready | a in [{selected_a_values[0]:.3f}, {selected_a_values[-1]:.3f}] nm",
            progress_log,
        )
        bayes_t0 = time.time()
        bayes_results, bo_candidates, bo_eval = run_bayesian_optimization(
            (float(selected_a_values[0]), float(selected_a_values[-1])),
            r_values,
            d_values,
            selected_window_results,
            active_family,
            args,
            progress_log=progress_log,
        )
        record_stage_time(stage_times, "bayes_opt", "Bayes opt", time.time() - bayes_t0, progress_log)
        if bayes_results:
            combined_results.extend(bayes_results)
            candidate_pool = merge_unique_candidates(
                bo_candidates,
                candidate_pool,
                args.max_refine_centers,
            )
    else:
        if args.bo_calls > 0 and not bo_available:
            report_progress(
                "scikit-optimize is not available in this environment; skipping Bayes optimization. "
                "Install it with `python -m pip install scikit-optimize` to enable BO.",
                progress_log,
            )
            if SKOPT_IMPORT_ERROR:
                report_progress(f"skopt import error: {SKOPT_IMPORT_ERROR}", progress_log)
        record_stage_time(stage_times, "bayes_opt", "Bayes opt", 0.0, progress_log)

    refine_t0 = time.time()
    refined_results = refine_candidates_3d(
        candidate_pool,
        selected_a_values,
        r_values,
        d_values,
        active_family,
        args,
        progress_log,
    )
    record_stage_time(stage_times, "refinement", "refinement", time.time() - refine_t0, progress_log)
    if refined_results:
        candidate_pool = (
            select_top_scan_candidates(
                refined_results,
                max_candidates=args.max_refine_centers,
                args=args,
            )
            or candidate_pool
        )
    candidate_pool = enrich_results_with_purcell(
        candidate_pool,
        active_family,
        args,
        len(candidate_pool),
        progress_log=progress_log,
        stage_label="candidate_purcell",
    )

    example, path_results, verification_elapsed = verify_candidates(
        candidate_pool,
        active_family,
        args,
        progress_log,
    )
    record_stage_time(stage_times, "verification", "validation", verification_elapsed, progress_log)

    final_results = merge_unique_scan_results(refined_results, combined_results) if refined_results else combined_results
    final_results = enrich_results_with_purcell(
        final_results,
        active_family,
        args,
        min(max(1, int(args.purcell_top_evals)), max(1, len(candidate_pool))),
        progress_log=progress_log,
        stage_label="final_purcell",
    )
    if example is None:
        example, path_results = build_fallback_example(
            final_results,
            active_family,
            args,
            progress_log,
        )

    robust_summaries, robust_best, robust_elapsed = run_fabrication_robustness_screen(
        candidate_pool if candidate_pool else final_results,
        active_family,
        args,
        progress_log,
    )
    record_stage_time(stage_times, "robustness", "robustness", robust_elapsed, progress_log)

    family_lookup = {active_family.key: active_family}
    figures: List[plt.Figure] = []
    r_slice_values = pick_best_r_slice(final_results, r_values, args, example)
    phase_points_by_r = build_phase_points_by_r(
        final_results if final_results else selected_window_results,
        r_slice_values,
    )
    figures.append(
        plot_phase_diagrams_r_slices(
            os.path.join(outdir, "bic_phase_diagram.png"),
            phase_points_by_r,
            example,
            family_lookup,
            args,
        )
    )
    if args.bo_calls > 0:
        figures.append(
            plot_bo_pred_vs_true(
                os.path.join(outdir, "bo_pred_vs_true.png"),
                bo_eval,
                args,
            )
        )
        figures.append(
            plot_bo_error_vs_iter(
                os.path.join(outdir, "bo_error_vs_iter.png"),
                bo_eval,
                args,
            )
        )
        figures.append(
            plot_bo_ci_coverage(
                os.path.join(outdir, "bo_ci_coverage.png"),
                bo_eval,
                args,
            )
        )
    figures.append(
        plot_candidate_purcell(
            os.path.join(outdir, "candidate_purcell.png"),
            candidate_pool if candidate_pool else final_results,
            args,
        )
    )
    if example is not None:
        figures.append(
            plot_example_dispersion(
                os.path.join(outdir, "bic_example_gamma_dispersion.png"),
                example,
                path_results,
                family_lookup,
                args,
            )
        )

    record_stage_time(stage_times, "total", "total", time.time() - t0, progress_log)

    save_scan_results_csv(os.path.join(outdir, "scan_results.csv"), final_results)
    if robust_summaries:
        save_robustness_csv(os.path.join(outdir, "robustness_candidates.csv"), robust_summaries)
    save_summary(
        os.path.join(outdir, "bic_search_summary.txt"),
        args,
        acceleration,
        float(selected_a_values[0]) if len(selected_a_values) else math.nan,
        float(selected_a_values[-1]) if len(selected_a_values) else math.nan,
        final_results,
        example,
        stage_times,
        progress_log,
        bo_eval=bo_eval,
        robust_summaries=robust_summaries,
        robust_best=robust_best,
    )

    print(f"Finished in {stage_times['total']:.2f} s")
    print(f"Outputs saved in: {outdir}")
    if example is not None:
        example_tag = "validated" if bool(example.get("validated", False)) else "fallback"
        purcell_text = ""
        if np.isfinite(example.get("purcell_factor_est", math.nan)):
            purcell_text = f", Purcell ≈ {example['purcell_factor_est']:.3e}"
        print(
            f"Selected example ({example_tag}): "
            f"a = {example['a_nm']:.3f} nm, hole radius ratio r/a = {example['r_over_a']:.4f}, "
            f"h = {example['h_nm']:.3f} nm, {example['pol']} lambda = {example['lambda_nm']:.3f} nm"
            f"{purcell_text}"
        )
    else:
        print("No example was found inside the selected scan windows for the selected polarization.")
    if robust_best is not None:
        print(
            "Robust best: "
            f"a = {robust_best.a_nm:.3f} nm, hole radius ratio r/a = {robust_best.r_over_a:.4f}, "
            f"h = {robust_best.h_nm:.3f} nm, yield = {robust_best.yield_fraction:.3f}, "
            f"robust score = {robust_best.robust_score:.3f}"
        )

    if not args.no_show_plots and figures:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    mp.freeze_support()
    main_run()
