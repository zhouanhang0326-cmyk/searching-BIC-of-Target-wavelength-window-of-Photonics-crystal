# searching-BIC-of-Target-wavelength-window-of-Photonics-crystal
BIC Photonic Crystal Cavity Designer
Overview

This project is a Bayesian-optimization-assisted photonic-crystal cavity design program built on top of Legume, the guided-mode expansion (GME) package developed within the fancompute / Fan Group ecosystem. Legume is a Python implementation of guided-mode expansion for photonic-crystal slabs and multilayer structures, and it also includes plane-wave expansion tools for purely 2D periodic systems.

The code is designed for fast screening of square-lattice photonic-crystal slab resonators intended for BIC or quasi-BIC laser-cavity design. It searches the geometric parameter space defined by the period a, the hole radius ratio r/a, and the slab thickness ratio d/a, and it tries to place a target BIC into a user-defined wavelength window while keeping the resonance sufficiently isolated from nearby modes for practical single-mode cavity operation. In the current implementation, that “FSR requirement” is enforced through a mode-isolation criterion: the target wavelength window is rewarded when it contains exactly one mode and exactly one BIC candidate, which serves as an engineering proxy for maintaining usable spectral spacing.

At present, this workflow supports non-dispersive, constant-index material models only. In other words, the slab and hole/background media are specified through fixed refractive indices (n_slab, n_hole) rather than wavelength-dependent n(λ) or k(λ). This makes the program well suited to preliminary photonic-crystal design using fixed-index approximations of materials such as Si3N4 in the visible, Si in the infrared, and TiO2 over a limited spectral range. Materials such as VO2 can also be explored at the constant-index approximation level, but the present code does not yet include a native dispersive or lossy material model.

Design Philosophy

The core idea of this repository is to accelerate BIC screening without sacrificing the physical logic of the cavity search. Instead of performing an expensive dense scan of the full parameter space at high fidelity from the beginning, the code uses a staged workflow:

Coarse Γ-point screening
Adaptive period-window expansion
Bayesian optimization over the promising region
Local refinement around shortlisted candidates
Near-Γ validation along a short momentum-space path
Optional Purcell-factor estimation
Optional fabrication-tolerance robustness screening

This staged design significantly reduces the number of expensive eigenmode calculations needed to identify a usable BIC cavity candidate. The script explicitly implements Bayesian optimization through skopt.Optimizer, and it records BO prediction error, uncertainty coverage, and refinement history as part of the output.

Main Features
1. Fast BIC screening in a target wavelength window

The program solves Γ-point modes for each scan point, converts normalized frequencies into physical wavelengths, and checks whether the geometry supports a candidate resonance inside the user-defined wavelength interval. By default, the code is organized around a target window such as 520–540 nm, but any wavelength range can be specified through command-line arguments. It computes the number of resonances in the window and the number of low-loss, high-Q candidates in that same window.

2. Effective FSR enforcement through mode isolation

The code does not compute a full cavity FSR spectrum in the conventional Fabry–Pérot sense. Instead, it applies an effective mode-spacing constraint by strongly favoring parameter points for which the target wavelength window contains one and only one mode and one and only one BIC candidate. This is an appropriate practical criterion for photonic-crystal laser-cavity pre-design, where the main goal is often to isolate a target high-Q resonance from competing modes.

3. Bayesian optimization for faster candidate discovery

After an initial coarse bootstrap scan, the program switches to Gaussian-process-based Bayesian optimization using scikit-optimize. This allows the search to concentrate computational effort on parameter regions that are more likely to host a useful BIC candidate, rather than spending equal time on obviously poor geometries. The scikit-optimize package is a sequential model-based optimization toolbox; the current latest PyPI release is 0.10.2.

4. Near-Γ validation instead of Γ-point-only ranking

A geometry that looks promising exactly at Γ is not always stable when the in-plane wavevector moves slightly away from Γ. To address this, the code re-evaluates selected candidates on a short signed path around Γ and uses that information as part of the ranking. This is especially important for practical BIC cavity design, because many “good-looking” Γ-point solutions fail once a small momentum-space neighborhood is considered.

5. Approximate Purcell factor and mode volume estimation

For the best shortlisted candidates, the code can estimate an approximate unit-cell Purcell factor and an approximate mode volume using the field distribution returned by Legume. These quantities are not used as the primary optimization objective; instead, they are treated as higher-level cavity-quality indicators reported after the main search.

6. Fabrication-tolerance robustness screening

The program can apply a small Monte Carlo fabrication-robustness screen around the final candidates by perturbing the period, hole radius, and slab thickness and then recomputing the Γ-point screening metrics. This yields a simple measure of robustness, including a nominal score, a mean score, a standard deviation, a lower quantile, and an estimated yield fraction.

Supported Material Scope

This repository currently targets constant-index dielectric photonic-crystal slab design. Typical use cases include:

Si3N4 photonic-crystal slabs in the visible
Si photonic-crystal slabs in the near-infrared / infrared
TiO2 photonic-crystal slabs in visible and near-visible bands
VO2 only under a fixed-index approximation over a narrow design window

Because the material model is currently non-dispersive, the code is most reliable when the design wavelength range is narrow enough that a constant effective refractive index is a good approximation. For strongly dispersive or absorptive materials, the next natural extension is to add wavelength-dependent n(λ) and k(λ) support.

Software Requirements

The script directly imports:

legume
numpy
matplotlib
scikit-optimize
optionally SciencePlots

A practical recommended stack is:

Python 3.10 or 3.11
legume-gme==1.0.2
numpy>=1.20.3
matplotlib>=3.6
scikit-optimize==0.10.2
SciencePlots==2.2.1 (optional, for publication-style figures)

The latest PyPI package pages currently list legume-gme 1.0.2, scikit-optimize 0.10.2, and SciencePlots 2.2.1. SciencePlots also requires an explicit import scienceplots before plt.style.use(...) in modern versions.

Installation
conda create -n bic-legume python=3.11
conda activate bic-legume

pip install legume-gme==1.0.2
pip install numpy matplotlib scikit-optimize==0.10.2
pip install SciencePlots==2.2.1

If you do not need the enhanced plotting style, SciencePlots can be omitted and the script will fall back to standard Matplotlib styling. The current code already handles that case by wrapping import scienceplots in a try/except block.

Usage

The script exposes a command-line interface for selecting the design wavelength window, scan mode, polarization family, geometry range, Bayesian optimization budget, refinement settings, validation settings, and output directory. Its CLI description explicitly states that it performs a physically constrained coarse scan near Γ, then optional refinement, and then a short Γ-centered validation path.

A typical run looks like this:

python findBIC.py \
  --project-name findSi3N4BIC_V4 \
  --range1-nm 520 \
  --range2-nm 540 \
  --scan-mode a_d \
  --pol TM \
  --n-slab 2.0 \
  --n-hole 1.49 \
  --fixed-r-over-a 0.32 \
  --d-min 0.10 \
  --d-max 2.50 \
  --d-points 30 \
  --a-points 20 \
  --bo-calls 45 \
  --bo-random-starts 10 \
  --validate-gmax 5.5

For a lightweight scouting run, the code also includes a --fast-test mode, which automatically reduces scan density, BO budget, and validation settings for quick exploration before a higher-accuracy run.

What the Code Computes

For each scan point, the program can compute:

Γ-point eigenfrequencies
Γ-point imaginary eigenfrequencies
approximate quality factor Q
physical wavelength mapped from normalized frequency
number of modes inside the target wavelength window
number of BIC candidates inside the target wavelength window
a continuous search score used for ranking and BO
a near-Γ validation score
approximate Purcell factor
approximate mode volume
Monte Carlo robustness statistics for final candidates

This makes the repository more than a simple band-structure script: it is a screen-search-validate pipeline for BIC-based photonic-crystal cavity design.

How to Improve Numerical Accuracy

For practical use, the most important accuracy controls are:

--gmax
--validate-gmax
--numeig-gamma
--numeig-path
--path-points-per-side
--near-gamma-points-per-side

The script already separates screening fidelity from final validation fidelity by using a moderate gmax during screening and a larger validate-gmax during the final near-Γ validation. In general, increasing gmax is the most direct way to improve the Legume basis accuracy, while increasing the number of path samples improves the reliability of near-Γ validation.

A recommended practical workflow is:

Use moderate gmax and moderate scan density for global search.
Increase validate-gmax for the final shortlist.
Increase path-points-per-side when validating the final candidate.
Repeat the final run with stricter settings to confirm convergence.

For future development, an even more rigorous strategy would be to validate the selected candidate in a 2D neighborhood around Γ, rather than only on a single short path.

Output Files

A successful run produces a timestamped output directory containing numerical results, summary files, and figures. The script explicitly saves:

scan_results.csv
robustness_candidates.csv (if robustness screening is enabled)
bic_search_summary.txt

and figure files such as:

bic_phase_diagram.png
bo_pred_vs_true.png
bo_error_vs_iter.png
bo_ci_coverage.png
candidate_purcell.png
bic_example_gamma_dispersion.png

These outputs are generated directly by the main workflow after the search, validation, and robustness stages complete.

Figure Types
BIC phase diagram

This figure shows the scan results in the selected parameter plane. It distinguishes regions with no target-window Γ-point BIC, regions where a Γ-point BIC exists but the target window is not spectrally isolated, and regions where the design window contains exactly one target mode and one BIC candidate.

BO surrogate diagnostics

The repository can export three BO diagnostic plots:

predicted score vs. true score
absolute prediction error vs. BO iteration
95% confidence-interval coverage vs. BO iteration

These are useful for checking whether the Bayesian surrogate is learning the design landscape sensibly.

Candidate Purcell comparison

This figure compares the best shortlisted candidates through an approximate unit-cell Purcell-factor estimate.

Example near-Γ dispersion and field map

The final example figure combines a near-Γ dispersion plot, colored by log10(Q), with a representative unit-cell electric-field map for the selected mode.

Current Limitations

This repository should be understood as a fast electromagnetic pre-design tool for BIC cavity screening, not yet as a full laser simulator. The current implementation does not include:

wavelength-dependent dispersive material fitting
material absorption k(λ)
gain or threshold modeling
carrier dynamics
thermal effects
nonlinear laser dynamics
full finite-size cavity modeling beyond the present periodic-cell workflow

It is therefore best suited to the following design question:

Which square-lattice photonic-crystal slab geometry places an isolated, high-Q BIC candidate inside my target wavelength window, with reasonable fabrication tolerance?

That is exactly the task this code is designed to accelerate.

Citation and Acknowledgment

If you use this repository in academic work, please acknowledge:

Legume, the guided-mode expansion package from the fancompute / Fan Group ecosystem for photonic-crystal slab eigenmode calculations.
scikit-optimize for Bayesian optimization
