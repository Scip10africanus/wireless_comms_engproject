"""
Bayesian Optimization Parameter Optimizer

Optimises the same three genes as ``ga_parameter_optimizer.m`` (DCT path):

  * ``K``        — zigzag split, **[5 .. 32]**
  * ``Qb_idx`` — base-layer quantisation index, **[0 .. 3]**
  * ``Qe_idx`` — enhancement-layer quantisation index, **[0 .. 3]**

MCS is **not** in the search space (handled by rules + ceiling in the live
system); the Python fitness is a port of MATLAB ``calc_fitness`` /
``regime_weights`` for offline BO vs GA comparison.

Uses a Gaussian Process surrogate model (via scikit-optimize) to learn the
fitness landscape and pick promising points to evaluate next — rather than
evolving a random population like the GA does.

Designed for direct GA vs BO comparison in the project report.

Fair comparison mode (default for ``compare_bo_vs_ga``):
  * **Fair start (default):** ``fair_start=True`` gives **cold BO** (no seed
    ``tell``) and a **uniformly random initial GA population** (not
    ``[20,0,0]``). BO's skopt ``random_state`` and the GA init RNG both use
    ``ga_init_seed + 1000 * SNR`` per sweep point so neither side reuses the
    same pseudo-random trajectory at every SNR. Set ``fair_start=False`` or
    env ``BO_GA_MATLAB_MATCH=1`` for MATLAB-style GA seed and optional BO warm
    start (``BO_GA_WARM_BO=1``).
  * **Matched budget:** BO runs exactly ``matched_fitness_budget()`` coded
    fitness evaluations (same count as the GA population-eval budget).
    Full run = 640; with env ``BO_GA_QUICK=1`` this drops to 180 for faster
    figures.
  * **Convergence plot:** both curves are **cumulative best fitness vs
    evaluation index** (same x-axis semantics).
  * The Python GA replica uses the same tournament rule as MATLAB
    ``evolve``: parents are chosen by **re-using** the current generation's
    fitness vector (no duplicate ``calc_fitness`` calls for tournaments).
"""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

try:
    from skopt import Optimizer
    from skopt.space import Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# --- GA defaults (must stay in sync with _run_ga) ---
GA_ELITE_COUNT = 4
GA_SEED_CHROMOSOME = [20, 0, 0]  # align with ga_parameter_optimizer.m defaults


def _is_quick_compare():
    """Set ``BO_GA_QUICK=1`` for smaller GA/BO budgets (minutes vs hours)."""
    return os.environ.get("BO_GA_QUICK", "").lower() in ("1", "true", "yes", "y")


def ga_pop_size():
    return 20 if _is_quick_compare() else 40


def ga_n_generations():
    return 8 if _is_quick_compare() else 15


def matched_fitness_budget():
    """Population fitness evaluations per GA run (incl. final pop scan)."""
    p, g = ga_pop_size(), ga_n_generations()
    return p * g + p


# Full-run reference (non-quick) for captions / docs
MATCHED_FITNESS_BUDGET_FULL = 40 * 15 + 40


# ---------------------------------------------------------------------------
# Fitness — port of ga_parameter_optimizer.m :: calc_fitness + regime_weights
# ---------------------------------------------------------------------------


def default_prev_quality():
    """Same as missing chCond fields in MATLAB (NaN telemetry)."""
    return {'ssim': np.nan, 'psnr': np.nan, 'enhFailRatio': np.nan}


def regime_weights(snr_dB, prev_ssim):
    """Port of ``regime_weights`` in ``ga_parameter_optimizer.m``."""
    W_CRISIS = np.array([2.5, 0.0, 1.0], dtype=float)
    W_STRESSED = np.array([2.2, 0.3, 0.8], dtype=float)
    W_COMFORTABLE = np.array([2.15, 0.38, 0.42], dtype=float)
    W_SURPLUS = np.array([1.65, 0.42, 0.32], dtype=float)

    has_prev = not np.isnan(prev_ssim)

    if has_prev and prev_ssim < 0.85:
        return W_CRISIS[0], W_CRISIS[1], W_CRISIS[2]
    if has_prev and prev_ssim < 0.93 and snr_dB >= 20.0:
        alpha = (prev_ssim - 0.85) / 0.08
        alpha = float(np.clip(alpha, 0.0, 1.0))
        w = (1.0 - alpha) * W_CRISIS + alpha * W_STRESSED
        return float(w[0]), float(w[1]), float(w[2])

    surplus_ok = has_prev and prev_ssim >= 0.97
    blend = 2.0

    if snr_dB < (12.0 - blend):
        w = W_CRISIS
    elif snr_dB < (12.0 + blend):
        t = (snr_dB - (12.0 - blend)) / (2.0 * blend)
        w = (1.0 - t) * W_CRISIS + t * W_STRESSED
    elif snr_dB < (20.0 - blend):
        w = W_STRESSED
    elif snr_dB < (20.0 + blend):
        t = (snr_dB - (20.0 - blend)) / (2.0 * blend)
        w = (1.0 - t) * W_STRESSED + t * W_COMFORTABLE
    elif snr_dB < (32.0 - blend):
        w = W_COMFORTABLE
    elif snr_dB < (32.0 + blend):
        if surplus_ok:
            t = (snr_dB - (32.0 - blend)) / (2.0 * blend)
            w = (1.0 - t) * W_COMFORTABLE + t * W_SURPLUS
        else:
            w = W_COMFORTABLE
    else:
        w = W_SURPLUS if surplus_ok else W_COMFORTABLE

    return float(w[0]), float(w[1]), float(w[2])


def calc_fitness(chr, snr_dB, prev_quality=None, current_k=20):
    """
    Port of MATLAB ``calc_fitness`` for chromosome ``[K, Qb_idx, Qe_idx]``.

    Parameters
    ----------
    chr : sequence of 3 ints
    snr_dB : float
    prev_quality : dict, optional
        Keys ``ssim``, ``psnr``, ``enhFailRatio`` (use NaN if unknown).
    current_k : int
        Prior frame K for stability penalty (MATlab ``currentK``).
    """
    if prev_quality is None:
        prev_quality = default_prev_quality()

    K = int(chr[0])
    qb_idx = int(chr[1])
    qe_idx = int(chr[2])
    k_ratio = K / 32.0
    q_sum = qb_idx + qe_idx

    q_dist_base = (qb_idx + 1) / 4.0
    q_dist_enh = (qe_idx + 1) / 4.0
    q_score = 1.0 - 0.5 * (k_ratio * q_dist_base + (1.0 - k_ratio) * q_dist_enh)

    comp_score = (qb_idx + qe_idx) / 6.0 + 0.25 * (1.0 - k_ratio)

    rician_penalty_db = 4.0
    eff_snr = float(snr_dB) - rician_penalty_db
    if eff_snr < 10.0:
        base_reliability = 0.5
    elif eff_snr < 20.0:
        base_reliability = 0.5 + 0.5 * (eff_snr - 10.0) / 10.0
    else:
        base_reliability = 1.0
    prot_score = (
        0.7 * k_ratio * base_reliability + 0.3 * (1.0 - k_ratio) * base_reliability
    )

    k_pen = 0.015 * ((K - 20) ** 2) / 10.0
    stability_pen = 0.02 * abs(K - int(current_k))

    prev_ssim = float(prev_quality.get('ssim', np.nan))
    prev_psnr = float(prev_quality.get('psnr', np.nan))
    enh_fail = prev_quality.get('enhFailRatio', np.nan)
    enh_fail = float(enh_fail) if enh_fail is not None else np.nan

    w_q, w_c, w_p = regime_weights(float(snr_dB), prev_ssim)
    f = w_q * q_score + w_c * comp_score + w_p * prot_score - k_pen - stability_pen

    if not np.isnan(prev_ssim):
        if prev_ssim < 0.95:
            f -= 1.2 * (0.95 - prev_ssim)
        f += 0.4 * (prev_ssim - 0.93)
        if prev_ssim < 0.85:
            f += 0.2 * (K - 20) / 20.0

    if not np.isnan(enh_fail) and enh_fail > 0.5:
        f -= 0.3 * (1.0 - k_ratio)

    if not np.isnan(prev_psnr) and prev_psnr < 40.0:
        f -= 0.5 * q_sum
    if np.isnan(prev_ssim) and np.isnan(prev_psnr):
        f -= 0.45 * q_sum
    if not np.isnan(prev_ssim) and prev_ssim >= 0.96 and q_sum > 0:
        f -= 0.15 * q_sum * (prev_ssim - 0.95) / 0.05

    return float(f)


# ---------------------------------------------------------------------------
# Bayesian Optimizer class
# ---------------------------------------------------------------------------

class BayesianParameterOptimizer:
    """
    Bayesian optimisation over ``[K, Qb_idx, Qe_idx]`` (same as MATLAB GA).

    After each evaluation the GP surrogate is updated; the next point is
    chosen via the scikit-optimize acquisition function (EI by default).
    """

    SPACE = [
        Integer(5, 32, name='K'),
        Integer(0, 3, name='Qb_idx'),
        Integer(0, 3, name='Qe_idx'),
    ]

    def __init__(self, n_initial=8, n_calls=30, acq_func='EI', verbose=False, random_state=42):
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize not found.\n"
                "Install it with:  pip install scikit-optimize"
            )
        self.n_initial = n_initial
        self.n_calls   = n_calls
        self.acq_func  = acq_func
        self.verbose   = verbose
        self._random_state = random_state

        self._optimizer = Optimizer(
            dimensions=self.SPACE,
            base_estimator='GP',        # Gaussian Process surrogate
            acq_func=acq_func,
            n_initial_points=n_initial,
            random_state=random_state,
        )

        # History for plotting / comparison
        self.eval_history = []   # list of {'params', 'fitness', 'call_idx'}
        self.best_params  = None
        self.best_fitness = -np.inf

    # ------------------------------------------------------------------
    def optimize(self, snr_db, seed_params=None, prev_quality=None, current_k=20):
        """
        Run Bayesian optimisation and return the best parameter set.

        Parameters
        ----------
        snr_db : float
            Channel SNR in dB (same meaning as ``chCond.snr_dB``).
        seed_params : list of 3 ints, optional
            ``[K, Qb_idx, Qe_idx]`` warm-start (first ``tell``).
        prev_quality : dict, optional
            ``ssim``, ``psnr``, ``enhFailRatio`` for measured-feedback terms.
        current_k : int
            Prior-frame ``K`` for the stability penalty.
        """
        if prev_quality is None:
            prev_quality = default_prev_quality()

        t0 = time.perf_counter()

        if seed_params is not None:
            seed = [
                int(np.clip(seed_params[i], self.SPACE[i].low, self.SPACE[i].high))
                for i in range(3)
            ]
            seed_fit = calc_fitness(seed, snr_db, prev_quality, current_k)
            self._optimizer.tell(seed, -seed_fit)   # skopt minimises
            self._record(seed, seed_fit, 0)

        for call_idx in range(1, self.n_calls + 1):
            suggestion = self._optimizer.ask()
            fitness = calc_fitness(suggestion, snr_db, prev_quality, current_k)
            self._optimizer.tell(suggestion, -fitness)   # negate → minimise
            self._record(suggestion, fitness, call_idx)

            if self.verbose:
                p = suggestion
                print(
                    f"  BO call {call_idx:2d}: "
                    f"K={p[0]:2d} Qb={p[1]} Qe={p[2]}  fit={fitness:.4f}"
                )

        duration = time.perf_counter() - t0

        best_p = self.best_params
        metrics = {
            'fitness_history': [e['fitness'] for e in self.eval_history],
            'best_fitness':    self.best_fitness,
            'n_calls':         self.n_calls,
            'total_evals':     len(self.eval_history),
            'duration_s':      duration,
        }
        return best_p, metrics

    def _record(self, params, fitness, call_idx):
        entry = {
            'call_idx': call_idx,
            'fitness': fitness,
            'params': {
                'K': int(params[0]),
                'Qb_idx': int(params[1]),
                'Qe_idx': int(params[2]),
            },
        }
        self.eval_history.append(entry)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_params  = entry['params'].copy()

    def reset(self):
        """Reset the optimizer (call between frames if desired)."""
        self.__init__(
            self.n_initial,
            self.n_calls,
            self.acq_func,
            self.verbose,
            self._random_state,
        )


# ---------------------------------------------------------------------------
# Standalone comparison: BO vs GA on a sweep of SNR values
# ---------------------------------------------------------------------------

def _coded_evals_until_final_best(per_eval_or_cumulative, final_best=None):
    """
    Smallest 1-based coded-eval index i such that best-so-far >= ``final_best``.

    ``per_eval_or_cumulative`` is either raw fitness per evaluation (BO
    ``fitness_history``) or a non-decreasing cumulative-best series (GA).
    """
    arr = np.asarray(per_eval_or_cumulative, dtype=float)
    if arr.size == 0:
        return 0
    cum = np.maximum.accumulate(arr)
    fb = float(final_best) if final_best is not None else float(cum[-1])
    tol = max(1e-9, 1e-7 * max(1.0, abs(fb)))
    hit = np.flatnonzero(cum >= fb - tol)
    return int(hit[0]) + 1 if hit.size else int(arr.size)


def compare_bo_vs_ga(
    snr_range=None,
    n_calls_bo=None,
    save_plot=True,
    fair_budget=True,
    seed_chromosome=None,
    use_bo_seed=False,
    fair_start=True,
    use_ga_seed=True,
    ga_init_seed=42,
    prev_quality=None,
    current_k=20,
):
    """
    Run both the Bayesian Optimizer and the GA over a range of SNR values
    and produce a comparison figure.

    Parameters
    ----------
    n_calls_bo : int or None
        BO ``ask``/``tell`` iterations after any seed. If ``None`` and
        ``fair_budget`` is True, uses ``matched_fitness_budget() - 1`` when
        ``use_bo_seed`` (seed ``tell`` + loop = GA budget) else
        ``matched_fitness_budget()`` (cold BO, same total eval count as GA).
    fair_budget : bool
        When True (default) and ``n_calls_bo is None``, set ``n_calls_bo`` so
        total BO coded evals equals the GA population-eval budget (``+1`` when
        ``use_bo_seed`` because of the extra seed ``tell``).
    seed_chromosome : list or None
        Used only if ``use_bo_seed`` is True: first ``tell`` to BO before
        surrogate-guided ``ask``/``tell``. Defaults to ``GA_SEED_CHROMOSOME``.
    use_bo_seed : bool
        If True, BO is warm-started with ``seed_chromosome``. Ignored when
        ``fair_start`` is True (cold BO is forced).
    fair_start : bool
        If True (default), run a **fair start** benchmark: cold BO and a fully
        random initial GA population (uniform in each gene range). ``use_bo_seed``
        and ``use_ga_seed`` are then ignored.
    use_ga_seed : bool
        If True (default when ``fair_start`` is False), initialise the GA like
        ``ga_parameter_optimizer.m`` (first individual = ``GA_SEED_CHROMOSOME``,
        others perturbed). If False, initial population is i.i.d. uniform in
        ``[K,Qb,Qe]`` bounds (``ga_init_seed`` fixes reproducibility).
    ga_init_seed : int
        Passed to ``numpy.random.default_rng`` for the random initial population
        when ``use_ga_seed`` is False.
    prev_quality : dict or None
        Measured feedback (``ssim``, ``psnr``, ``enhFailRatio``); defaults to
        all-NaN like missing ``chCond`` fields in MATLAB.
    current_k : int
        Prior-frame ``K`` for the stability term (``curParams.K`` in MATLAB).

    Requires ga_parameter_optimizer.m to be accessible only if running inside
    MATLAB; here we use the pure-Python fitness function for both.
    """
    if fair_start:
        use_bo_seed = False
        use_ga_seed = False

    if snr_range is None:
        if _is_quick_compare():
            snr_range = [8, 14, 20, 26]
        else:
            snr_range = [5, 10, 15, 20, 25, 30]

    if prev_quality is None:
        prev_quality = default_prev_quality()
    else:
        prev_quality = {**default_prev_quality(), **prev_quality}

    if seed_chromosome is None:
        seed_chromosome = list(GA_SEED_CHROMOSOME)

    bo_seed_arg = seed_chromosome if use_bo_seed else None

    _mb = matched_fitness_budget()
    if n_calls_bo is None:
        if fair_budget:
            # Seed tell + asks == matched GA pop-eval budget (when use_bo_seed)
            n_calls_bo = _mb - 1 if use_bo_seed else _mb
        else:
            n_calls_bo = 30

    n_initial = int(min(25, max(8, n_calls_bo // 25)))

    bo_coded_evals = (1 + n_calls_bo) if use_bo_seed else n_calls_bo
    print("=" * 65)
    print("  Bayesian Optimization  vs  Genetic Algorithm")
    print("  Parameter space: [K, Qb_idx, Qe_idx]  (matches ga_parameter_optimizer.m)")
    bo_budget_note = (
        f"seed + {n_calls_bo} surrogate-guided asks"
        if use_bo_seed
        else f"{n_calls_bo} asks (no seed)"
    )
    print(f"  BO coded fitness evals: {bo_coded_evals}  ({bo_budget_note})")
    print(
        f"  GA population-eval budget (convergence plot): {_mb}  "
        f"(same x-axis only when BO evals == this value)"
    )
    if fair_start:
        print(
            "  NOTE: Fair start — cold BO + random GA init; BO/GA RNG keyed by "
            "ga_init_seed + 1000×SNR; left plot = evals until final best."
        )
    elif use_bo_seed:
        print("  NOTE: BO warm-started (GA seed chromosome as first tell).")
    else:
        print("  NOTE: BO cold start; left plot = evals until final best.")
    if not fair_start and not use_ga_seed:
        print("  NOTE: GA random init (MATLAB-style seed off).")
    if _is_quick_compare():
        print("  NOTE: BO_GA_QUICK=1 — smaller GA/BO budget for faster runs.")
    print("=" * 65)

    bo_results = {}
    ga_results = {}

    for snr in snr_range:
        # --- BO (same coded-eval count as GA population budget when fair) ---
        bo_rs = int(ga_init_seed) + int(snr) * 1000
        bo = BayesianParameterOptimizer(
            n_initial=n_initial,
            n_calls=n_calls_bo,
            acq_func='EI',
            verbose=False,
            random_state=bo_rs,
        )
        _, bo_m = bo.optimize(
            snr,
            seed_params=bo_seed_arg,
            prev_quality=prev_quality,
            current_k=current_k,
        )
        bo_results[snr] = bo_m
        bo_etb = _coded_evals_until_final_best(
            bo_m['fitness_history'], bo_m['best_fitness']
        )
        print(
            f"  SNR {snr:2d} dB | BO best={bo_m['best_fitness']:.4f} "
            f"evals-to-best={bo_etb}/{bo_m['total_evals']}  "
            f"({bo_m['duration_s']*1000:.1f} ms)"
        )

        # --- GA (Python replica of the MATLAB GA) ---
        ga_fit, ga_m = _run_ga(
            snr,
            prev_quality=prev_quality,
            current_k=current_k,
            use_ga_seed=use_ga_seed,
            ga_init_seed=int(ga_init_seed) + int(snr) * 1000,
        )
        ga_results[snr] = {'best_fitness': ga_fit, **ga_m}
        ga_etb = _coded_evals_until_final_best(
            ga_m['cumulative_best_pop_evals'], ga_fit
        )
        print(
            f"           | GA best={ga_fit:.4f}  evals-to-best={ga_etb}/"
            f"{ga_m['total_pop_evals']}  "
            f"fitness calls={ga_m['total_fitness_calls']}  "
            f"({ga_m['duration_ms']:.1f} ms)"
        )

    print()
    _plot_comparison(snr_range, bo_results, ga_results, save_plot, fair_start=fair_start)
    return bo_results, ga_results


# ---------------------------------------------------------------------------
# Pure-Python GA replica (matches ga_parameter_optimizer.m defaults)
# ---------------------------------------------------------------------------

def _run_ga(
    snr_db,
    pop_size=None,
    n_gen=None,
    mutation_rate=0.15,
    elite_count=GA_ELITE_COUNT,
    prev_quality=None,
    current_k=20,
    use_ga_seed=True,
    ga_init_seed=42,
):
    """Python replica of ``ga_parameter_optimizer.m`` (operators + fitness).

    Tournament parents are chosen using **indices into the current ``fit``**
    vector, matching MATLAB ``evolve`` (no duplicate fitness evaluations).

    When ``use_ga_seed`` is False, the **initial** population is drawn uniformly
    at random in each gene's bounds (reproducible via ``ga_init_seed``); later
    generations still use global ``numpy.random`` for crossover/mutation like
    the MATLAB script.
    """
    import time as _time

    if prev_quality is None:
        prev_quality = default_prev_quality()

    if pop_size is None:
        pop_size = ga_pop_size()
    if n_gen is None:
        n_gen = ga_n_generations()

    g_min = np.array([5, 0, 0], dtype=int)
    g_max = np.array([32, 3, 3], dtype=int)
    n_genes = 3

    def clamp(x):
        return np.clip(np.round(x), g_min, g_max).astype(int)

    fitness_calls = 0

    def eval_fitness(chrom):
        nonlocal fitness_calls
        fitness_calls += 1
        return calc_fitness(chrom, snr_db, prev_quality, current_k)

    t0 = _time.perf_counter()

    pop = np.zeros((pop_size, n_genes), dtype=int)
    if use_ga_seed:
        seed = np.array(GA_SEED_CHROMOSOME, dtype=float)
        pop[0] = clamp(seed)
        for i in range(1, pop_size):
            pop[i] = clamp(seed + np.round(np.random.randn(n_genes) * [3, 1, 1]))
    else:
        rng = np.random.default_rng(int(ga_init_seed))
        for j in range(n_genes):
            lo, hi = int(g_min[j]), int(g_max[j])
            pop[:, j] = rng.integers(lo, hi + 1, size=pop_size)

    fit_hist = []
    cumulative_best_pop = []
    pop_eval_fitness_series = []
    best_so_far_pop = -np.inf

    for _ in range(n_gen):
        fit = np.array([eval_fitness(pop[i]) for i in range(pop_size)], dtype=float)
        fit_hist.append(float(fit.max()))
        for i in range(pop_size):
            fv = float(fit[i])
            pop_eval_fitness_series.append(fv)
            best_so_far_pop = max(best_so_far_pop, fv)
            cumulative_best_pop.append(best_so_far_pop)

        order = np.argsort(fit)[::-1]
        new_pop = np.zeros_like(pop)
        new_pop[:elite_count] = pop[order[:elite_count]]

        for i in range(elite_count, pop_size):
            idx1 = np.random.choice(pop_size, min(3, pop_size), replace=False)
            idx2 = np.random.choice(pop_size, min(3, pop_size), replace=False)
            p1 = pop[idx1[np.argmax(fit[idx1])]]
            p2 = pop[idx2[np.argmax(fit[idx2])]]

            mask = np.random.rand(n_genes) > 0.5
            child = p1.copy()
            child[mask] = p2[mask]

            for g in range(n_genes):
                if np.random.rand() < mutation_rate:
                    child[g] += int(
                        np.round(np.random.randn() * (g_max[g] - g_min[g]) * 0.2)
                    )
            new_pop[i] = clamp(child)
        pop = new_pop

    fit = np.array([eval_fitness(pop[i]) for i in range(pop_size)], dtype=float)
    for i in range(pop_size):
        fv = float(fit[i])
        pop_eval_fitness_series.append(fv)
        best_so_far_pop = max(best_so_far_pop, fv)
        cumulative_best_pop.append(best_so_far_pop)

    best = pop[int(np.argmax(fit))]
    best_fit = float(fit.max())
    duration_ms = (_time.perf_counter() - t0) * 1000

    return best_fit, {
        'fitness_history': fit_hist,
        'cumulative_best_pop_evals': cumulative_best_pop,
        'pop_eval_fitness_series': pop_eval_fitness_series,
        'best_params': {
            'K': int(best[0]),
            'Qb_idx': int(best[1]),
            'Qe_idx': int(best[2]),
        },
        'total_pop_evals': len(cumulative_best_pop),
        'total_fitness_calls': fitness_calls,
        'duration_ms': duration_ms,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_comparison(snr_range, bo_results, ga_results, save=True, fair_start=True):
    """
    Original two-panel style (colors, boxed axes, legend wording), with a lower
    figure height and slightly thinner bars/lines than the first wide layout.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(14.2, 3.85), gridspec_kw={'wspace': 0.38}
    )

    # Left: evals until running best first reaches final best
    bo_etb = [
        _coded_evals_until_final_best(
            bo_results[s]['fitness_history'], bo_results[s]['best_fitness']
        )
        for s in snr_range
    ]
    ga_etb = [
        _coded_evals_until_final_best(
            ga_results[s]['cumulative_best_pop_evals'],
            ga_results[s]['best_fitness'],
        )
        for s in snr_range
    ]
    x_idx = np.arange(len(snr_range))
    bar_w = 0.28
    axes[0].bar(
        x_idx - bar_w / 2,
        ga_etb,
        bar_w,
        label='Genetic Algorithm',
        color='#2196F3',
        edgecolor='#1565C0',
        linewidth=0.45,
    )
    axes[0].bar(
        x_idx + bar_w / 2,
        bo_etb,
        bar_w,
        label='Bayesian Optimization',
        color='#FF5722',
        edgecolor='#E64A19',
        linewidth=0.45,
    )
    axes[0].set_xticks(x_idx)
    axes[0].set_xticklabels([str(s) for s in snr_range])
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('Coded evals until final best')
    axes[0].set_title(
        'Fair start: evals to best vs SNR'
        if fair_start
        else 'Evals until final best vs SNR'
    )
    axes[0].grid(True, axis='y', alpha=0.3)

    # Right: per-eval scatter + cumulative best (original palette, thinner strokes)
    mid_snr = snr_range[len(snr_range) // 2]
    bo_hist = list(bo_results[mid_snr]['fitness_history'])
    bo_cummax = np.maximum.accumulate(bo_hist)

    ga_cum = np.array(ga_results[mid_snr]['cumulative_best_pop_evals'], dtype=float)
    ga_raw = ga_results[mid_snr].get('pop_eval_fitness_series')
    if ga_raw is not None:
        ga_raw = np.array(ga_raw, dtype=float)
    m = min(len(bo_cummax), len(ga_cum))
    x = np.arange(m)
    mark_stride = max(1, m // 24) if m > 48 else 1

    axes[1].scatter(
        x,
        np.asarray(bo_hist, dtype=float)[:m],
        s=4,
        alpha=0.18,
        color='#FF5722',
        label='BO (each eval)',
        rasterized=True,
    )
    axes[1].plot(
        x,
        bo_cummax[:m],
        's-',
        markersize=2.8,
        markevery=mark_stride,
        label='BO (cumulative best)',
        color='#E64A19',
        linewidth=1.05,
        markeredgewidth=0.35,
    )
    if ga_raw is not None and len(ga_raw) >= m:
        axes[1].scatter(
            x,
            ga_raw[:m],
            s=4,
            alpha=0.18,
            color='#2196F3',
            label='GA (each pop eval)',
            rasterized=True,
        )
    axes[1].plot(
        x,
        ga_cum[:m],
        'o-',
        markersize=2.8,
        markevery=mark_stride,
        label='GA (cumulative best)',
        color='#1565C0',
        linewidth=1.05,
        markeredgewidth=0.35,
    )
    axes[1].set_xlabel('Coded fitness evaluation index')
    axes[1].set_ylabel('Fitness / best so far')
    axes[1].set_title(f'Convergence at {mid_snr} dB ({m} evals)')
    axes[1].grid(True, alpha=0.3)

    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    _leg = fig.legend(
        h0 + h1,
        l0 + l1,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=6.5,
        frameon=True,
        fancybox=False,
        edgecolor='0.78',
        handlelength=1.25,
        handletextpad=0.4,
        columnspacing=0.85,
        borderpad=0.35,
    )
    plt.tight_layout(pad=1.05, w_pad=2.0, rect=[0.0, 0.14, 1.0, 1.0])
    if save:
        plt.savefig(
            'bo_vs_ga_comparison.png',
            dpi=165,
            bbox_inches='tight',
            bbox_extra_artists=[_leg],
            pad_inches=0.1,
        )
        print("  ✓ Plot saved to bo_vs_ga_comparison.png")
    if os.environ.get("BO_GA_SHOW_PLOT", "").lower() in ("1", "true", "yes", "y"):
        plt.show()
    else:
        plt.close(fig)


def plot_bo_acquisition(snr_db=20, n_calls=30, prev_quality=None, current_k=20):
    """
    Visualise how BO explores the fitness landscape across evaluations
    for a fixed SNR — good figure for the report.
    """
    if prev_quality is None:
        prev_quality = default_prev_quality()
    bo = BayesianParameterOptimizer(n_initial=6, n_calls=n_calls, verbose=False)
    _, metrics = bo.optimize(
        snr_db,
        prev_quality=prev_quality,
        current_k=current_k,
    )

    hist = metrics['fitness_history']
    cummax = np.maximum.accumulate(hist)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(hist)), hist, alpha=0.4, color='#FF5722', label='Evaluation fitness')
    ax.plot(range(len(hist)), cummax, 'k-', linewidth=2, label='Best so far')
    ax.set_xlabel('BO Evaluation Call')
    ax.set_ylabel('Fitness')
    ax.set_title(f'BO acquisition trace ({snr_db} dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bo_acquisition_trace.png', dpi=150)
    print("  ✓ Plot saved to bo_acquisition_trace.png")
    if os.environ.get("BO_GA_SHOW_PLOT", "").lower() in ("1", "true", "yes", "y"):
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Quick usage demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Default: quick budget + fewer SNRs (~few minutes). Full paper run:
    #   BO_GA_FULL=1 BO_GA_QUICK=0 python3 bo_parameter_optimizer.py
    if os.environ.get("BO_GA_FULL", "").lower() not in ("1", "true", "yes", "y"):
        os.environ.setdefault("BO_GA_QUICK", "1")

    print("\n--- Single BO optimisation at SNR = 18 dB ---")
    print("    Chromosome [K, Qb_idx, Qe_idx]; fitness = ga_parameter_optimizer.m port")
    bo = BayesianParameterOptimizer(n_initial=8, n_calls=25, verbose=True)
    best, metrics = bo.optimize(
        snr_db=18.0,
        seed_params=[20, 0, 0],
        prev_quality=default_prev_quality(),
        current_k=20,
    )
    print(f"\nBest params : {best}")
    print(f"Best fitness: {metrics['best_fitness']:.4f}")
    print(f"Time        : {metrics['duration_s']*1000:.1f} ms")

    print("\n--- GA vs BO comparison (matched coded-eval budget) ---")
    if _is_quick_compare():
        print("    (BO_GA_QUICK=1: use BO_GA_FULL=1 for 640-eval × 7-SNR run; can take hours)")
    _matlab_match = os.environ.get("BO_GA_MATLAB_MATCH", "").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    _warm_bo = os.environ.get("BO_GA_WARM_BO", "").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    if _matlab_match:
        print(
            "    BO_GA_MATLAB_MATCH=1: MATLAB-style GA seed; "
            "BO warm start only if BO_GA_WARM_BO=1."
        )
    else:
        print("    Default: fair start (random GA init + cold BO). Set BO_GA_MATLAB_MATCH=1 to disable.")
    if _warm_bo and _matlab_match:
        print("    BO_GA_WARM_BO=1: BO warm-started with GA seed.")
    compare_bo_vs_ga(
        fair_start=not _matlab_match,
        use_bo_seed=_warm_bo if _matlab_match else False,
    )

    print("\n--- BO acquisition trace (for report figure) ---")
    plot_bo_acquisition(snr_db=20)
