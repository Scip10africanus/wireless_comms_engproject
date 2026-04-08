"""
Bayesian Optimization Parameter Optimizer

Optimises 3 TX parameters (Qb and Qe are fixed at 1 per team decision —
quantisation steps are not primary optimisation variables at this stage):
  K        - DCT coefficient split  [5 .. 60]
  mcs_base - Base layer MCS         [0 ..  7]
  mcs_enh  - Enhancement layer MCS  [0 ..  7]

Qb_idx = 1  (fixed — wavelet quantisation kept constant)
Qe_idx = 1  (fixed — wavelet quantisation kept constant)

Uses a Gaussian Process surrogate model (via scikit-optimize) to learn the
fitness landscape and pick promising points to evaluate next — rather than
evolving a random population like the GA does.

Designed for direct GA vs BO comparison in the project report.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

try:
    from skopt import Optimizer
    from skopt.space import Integer
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fitness function — mirrors ga_parameter_optimizer.m :: calc_fitness()
# Same MCS table, same BER model, same reward weights.
# ---------------------------------------------------------------------------

_MOD_ORDER  = [4,  4,  4,  16, 16,  16,  64,  64]
_CODE_RATE  = [0.5, 0.75, 1.0, 0.5, 0.75, 1.0, 0.67, 0.75]


def _approx_ber(snr_db, M, code_rate):
    """Rough M-QAM BER with coding gain (matches MATLAB approx_ber)."""
    from math import erfc, sqrt, log2
    snr_lin = 10 ** (snr_db / 10.0)
    k = log2(M)
    if M == 2:
        ber_u = 0.5 * erfc(sqrt(snr_lin))
    else:
        ber_u = (4 / k) * (1 - 1 / sqrt(M)) * 0.5 * erfc(
            sqrt(3 * k * snr_lin / (2 * (M - 1)))
        )
    return max(0.0, min(0.5, ber_u / (1.0 / (1.0 - code_rate + 0.1))))


# Fixed quantisation values per team decision (Angela, 2026-03-17):
# Qb and Qe are not optimisation variables — kept at 1 for both layers.
_FIXED_QB = 1
_FIXED_QE = 1


def calc_fitness(params, snr_db):
    """
    Analytical fitness identical to the MATLAB GA fitness function.
    params : [K, mcs_base, mcs_enh]  (integers) — Qb/Qe fixed at 1.
    Returns a scalar fitness value (higher = better).
    """
    K, mcs_b, mcs_e = int(params[0]), int(params[1]), int(params[2])
    Qb, Qe = _FIXED_QB, _FIXED_QE

    b_mod  = _MOD_ORDER[mcs_b];  b_rate = _CODE_RATE[mcs_b]
    e_mod  = _MOD_ORDER[mcs_e];  e_rate = _CODE_RATE[mcs_e]

    b_ber = _approx_ber(snr_db, b_mod, b_rate)
    e_ber = _approx_ber(snr_db, e_mod, e_rate)

    # Quality score
    bQ = max(0.0, 1.0 - 10 * b_ber)
    eQ = max(0.0, 1.0 -  5 * e_ber)
    qual_score = 0.7 * bQ + 0.3 * eQ

    # Throughput score
    k_ratio = K / 64.0
    import math
    bTh = math.log2(b_mod) * b_rate
    eTh = math.log2(e_mod) * e_rate
    th_score = (k_ratio * bTh + (1 - k_ratio) * eTh) / 6.0

    # Penalties (Qb/Qe penalty still included but now fixed/minimal)
    q_pen  = 0.01 * (Qb + Qe)
    k_pen  = 0.1 * (10 - K) if K < 10 else (0.05 * (K - 50) if K > 50 else 0)
    fail_pen = 2.0 if b_ber > 0.1 else (0.5 if b_ber > 0.05 else 0.0)

    return 2.0 * qual_score + 1.0 * th_score - q_pen - k_pen - fail_pen


# ---------------------------------------------------------------------------
# Bayesian Optimizer class
# ---------------------------------------------------------------------------

class BayesianParameterOptimizer:
    """
    Bayesian Optimization over the 5-parameter TX space.

    After each real or simulated evaluation the Gaussian Process surrogate
    is updated and the next candidate is chosen by maximising the acquisition
    function (Expected Improvement by default in skopt).

    Parameters
    ----------
    n_initial : int
        Random points used to warm-start the GP (exploration phase).
    n_calls : int
        Total evaluations budget (initial + BO-guided).
    acq_func : str
        Acquisition function: 'EI' (Expected Improvement),
        'PI' (Probability of Improvement), 'LCB' (Lower Confidence Bound).
    verbose : bool
    """

    # Search space: 3 parameters only (Qb/Qe fixed at 1 — not optimised)
    SPACE = [
        Integer(5,  60, name='K'),
        Integer(0,   7, name='mcs_base'),
        Integer(0,   7, name='mcs_enh'),
    ]

    def __init__(self, n_initial=8, n_calls=30, acq_func='EI', verbose=False):
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize not found.\n"
                "Install it with:  pip install scikit-optimize"
            )
        self.n_initial = n_initial
        self.n_calls   = n_calls
        self.acq_func  = acq_func
        self.verbose   = verbose

        self._optimizer = Optimizer(
            dimensions=self.SPACE,
            base_estimator='GP',        # Gaussian Process surrogate
            acq_func=acq_func,
            n_initial_points=n_initial,
            random_state=42,
        )

        # History for plotting / comparison
        self.eval_history = []   # list of {'params', 'fitness', 'call_idx'}
        self.best_params  = None
        self.best_fitness = -np.inf

    # ------------------------------------------------------------------
    def optimize(self, snr_db, seed_params=None):
        """
        Run Bayesian Optimization and return the best parameter set.

        Parameters
        ----------
        snr_db      : float  – current channel SNR in dB
        seed_params : list   – optional [K, Qb, Qe, mcs_base, mcs_enh] to
                               warm-start (seed as the first evaluation)

        Returns
        -------
        best_params : dict  {K, Qb, Qe, mcs_base, mcs_enh}
        bo_metrics  : dict  {fitness_history, best_fitness, n_calls, duration_s}
        """
        t0 = time.perf_counter()

        # Optionally pre-seed with the current known-good params
        # seed_params should be [K, mcs_base, mcs_enh] (3 values)
        if seed_params is not None:
            seed = [int(np.clip(seed_params[i],
                                self.SPACE[i].low,
                                self.SPACE[i].high)) for i in range(3)]
            seed_fit = calc_fitness(seed, snr_db)
            self._optimizer.tell(seed, -seed_fit)   # skopt minimises
            self._record(seed, seed_fit, 0)

        for call_idx in range(1, self.n_calls + 1):
            suggestion = self._optimizer.ask()
            fitness    = calc_fitness(suggestion, snr_db)
            self._optimizer.tell(suggestion, -fitness)   # negate → minimise
            self._record(suggestion, fitness, call_idx)

            if self.verbose:
                p = suggestion
                print(f"  BO call {call_idx:2d}: "
                      f"K={p[0]:2d} Qb={p[1]} Qe={p[2]} "
                      f"mcsB={p[3]} mcsE={p[4]}  fit={fitness:.4f}")

        duration = time.perf_counter() - t0

        best_p = self.best_params
        metrics = {
            'fitness_history': [e['fitness'] for e in self.eval_history],
            'best_fitness':    self.best_fitness,
            'n_calls':         self.n_calls,
            'duration_s':      duration,
        }
        return best_p, metrics

    def _record(self, params, fitness, call_idx):
        entry = {
            'call_idx': call_idx,
            'fitness':  fitness,
            'params':   {
                'K':        params[0],
                'Qb':       _FIXED_QB,   # fixed — not optimised
                'Qe':       _FIXED_QE,   # fixed — not optimised
                'mcs_base': params[1],
                'mcs_enh':  params[2],
            }
        }
        self.eval_history.append(entry)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_params  = entry['params'].copy()

    def reset(self):
        """Reset the optimizer (call between frames if desired)."""
        self.__init__(self.n_initial, self.n_calls, self.acq_func, self.verbose)


# ---------------------------------------------------------------------------
# Standalone comparison: BO vs GA on a sweep of SNR values
# ---------------------------------------------------------------------------

def compare_bo_vs_ga(snr_range=None, n_calls_bo=30, save_plot=True):
    """
    Run both the Bayesian Optimizer and the GA over a range of SNR values
    and produce a comparison figure.

    Requires ga_parameter_optimizer.m to be accessible only if running inside
    MATLAB; here we use the pure-Python fitness function for both.
    """
    if snr_range is None:
        snr_range = [5, 10, 15, 20, 25, 30]

    print("=" * 65)
    print("  Bayesian Optimization  vs  Genetic Algorithm")
    print("  Parameter space: K, Qb, Qe, mcs_base, mcs_enh")
    print("=" * 65)

    bo_results  = {}
    ga_results  = {}

    for snr in snr_range:
        # --- BO ---
        bo = BayesianParameterOptimizer(n_initial=8, n_calls=n_calls_bo,
                                         acq_func='EI', verbose=False)
        _, bo_m = bo.optimize(snr)
        bo_results[snr] = bo_m
        print(f"  SNR {snr:2d} dB | BO best={bo_m['best_fitness']:.4f} "
              f"in {n_calls_bo} evals  ({bo_m['duration_s']*1000:.1f} ms)")

        # --- GA (Python replica of the MATLAB GA) ---
        ga_fit, ga_m = _run_ga(snr)
        ga_results[snr] = {'best_fitness': ga_fit, **ga_m}
        print(f"           | GA best={ga_fit:.4f} "
              f"in {ga_m['total_evals']} evals ({ga_m['duration_ms']:.1f} ms)")

    print()
    _plot_comparison(snr_range, bo_results, ga_results, save_plot)
    return bo_results, ga_results


# ---------------------------------------------------------------------------
# Pure-Python GA replica (matches ga_parameter_optimizer.m defaults)
# ---------------------------------------------------------------------------

def _run_ga(snr_db, pop_size=40, n_gen=15, mutation_rate=0.15, elite_count=4):
    """Python replica of ga_parameter_optimizer.m for fair Python-vs-Python timing.
    Qb/Qe fixed at 1 — not part of the search."""
    import time as _time

    g_min = np.array([5, 0, 0])
    g_max = np.array([60, 7, 7])
    n_genes = 3  # K, mcs_base, mcs_enh only

    def clamp(x):
        return np.clip(np.round(x), g_min, g_max).astype(int)

    t0 = _time.perf_counter()

    # Initialise population (seed: K=20, mcs_base=2, mcs_enh=6)
    seed = np.array([20, 2, 6])
    pop  = np.zeros((pop_size, n_genes), dtype=int)
    pop[0] = clamp(seed)
    for i in range(1, pop_size):
        pop[i] = clamp(seed + np.round(np.random.randn(n_genes) * [5, 1, 1]))

    fit_hist = []

    for _ in range(n_gen):
        fit = np.array([calc_fitness(pop[i], snr_db) for i in range(pop_size)])
        fit_hist.append(fit.max())

        # Next generation
        order   = np.argsort(fit)[::-1]
        new_pop = np.zeros_like(pop)
        new_pop[:elite_count] = pop[order[:elite_count]]

        for i in range(elite_count, pop_size):
            # Tournament selection
            c1 = pop[np.random.choice(pop_size, min(3, pop_size), replace=False)]
            c2 = pop[np.random.choice(pop_size, min(3, pop_size), replace=False)]
            f1 = [calc_fitness(c, snr_db) for c in c1]
            f2 = [calc_fitness(c, snr_db) for c in c2]
            p1 = c1[np.argmax(f1)];  p2 = c2[np.argmax(f2)]

            # Crossover
            mask  = np.random.rand(n_genes) > 0.5
            child = p1.copy();  child[mask] = p2[mask]

            # Mutation
            for g in range(n_genes):
                if np.random.rand() < mutation_rate:
                    child[g] += int(np.round(np.random.randn() * (g_max[g] - g_min[g]) * 0.2))
            new_pop[i] = clamp(child)
        pop = new_pop

    fit = np.array([calc_fitness(pop[i], snr_db) for i in range(pop_size)])
    best = pop[fit.argmax()]
    best_fit = fit.max()
    duration_ms = (_time.perf_counter() - t0) * 1000

    return best_fit, {
        'fitness_history': fit_hist,
        'best_params': {
            'K': best[0], 'Qb': _FIXED_QB, 'Qe': _FIXED_QE,
            'mcs_base': best[1], 'mcs_enh': best[2]
        },
        'total_evals': pop_size * n_gen,
        'duration_ms': duration_ms,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_comparison(snr_range, bo_results, ga_results, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: best fitness vs SNR
    bo_fits = [bo_results[s]['best_fitness'] for s in snr_range]
    ga_fits = [ga_results[s]['best_fitness']  for s in snr_range]

    axes[0].plot(snr_range, ga_fits, 'o-', label='Genetic Algorithm',
                 linewidth=2, markersize=7, color='#2196F3')
    axes[0].plot(snr_range, bo_fits, 's--', label='Bayesian Optimization',
                 linewidth=2, markersize=7, color='#FF5722')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('Best Fitness Score')
    axes[0].set_title('GA vs BO — Fitness Across SNR Range')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: convergence at one mid SNR
    mid_snr = snr_range[len(snr_range) // 2]
    ga_hist = ga_results[mid_snr]['fitness_history']
    bo_hist = bo_results[mid_snr]['fitness_history']

    # Cumulative best for BO
    bo_cummax = np.maximum.accumulate(bo_hist)
    ga_x = np.linspace(0, len(bo_hist) - 1, len(ga_hist))

    axes[1].plot(np.arange(len(bo_hist)), bo_cummax, 's--',
                 label='BO (cumulative best)', color='#FF5722', linewidth=2)
    axes[1].plot(ga_x, ga_hist, 'o-',
                 label='GA (best per generation)', color='#2196F3', linewidth=2)
    axes[1].set_xlabel('Evaluation / Generation')
    axes[1].set_ylabel('Fitness Score')
    axes[1].set_title(f'Convergence Comparison at SNR = {mid_snr} dB')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('bo_vs_ga_comparison.png', dpi=150)
        print("  ✓ Plot saved to bo_vs_ga_comparison.png")
    plt.show()


def plot_bo_acquisition(snr_db=20, n_calls=30):
    """
    Visualise how BO explores the fitness landscape across evaluations
    for a fixed SNR — good figure for the report.
    """
    bo = BayesianParameterOptimizer(n_initial=6, n_calls=n_calls, verbose=False)
    _, metrics = bo.optimize(snr_db)

    hist = metrics['fitness_history']
    cummax = np.maximum.accumulate(hist)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(hist)), hist, alpha=0.4, color='#FF5722', label='Evaluation fitness')
    ax.plot(range(len(hist)), cummax, 'k-', linewidth=2, label='Best so far')
    ax.set_xlabel('BO Evaluation Call')
    ax.set_ylabel('Fitness')
    ax.set_title(f'Bayesian Optimization — Exploration vs Exploitation  (SNR={snr_db} dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bo_acquisition_trace.png', dpi=150)
    print("  ✓ Plot saved to bo_acquisition_trace.png")
    plt.show()


# ---------------------------------------------------------------------------
# Quick usage demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("\n--- Single BO optimisation at SNR = 18 dB ---")
    print("    (Qb=1, Qe=1 fixed; optimising K, mcs_base, mcs_enh)")
    bo = BayesianParameterOptimizer(n_initial=8, n_calls=25, verbose=True)
    best, metrics = bo.optimize(snr_db=18.0, seed_params=[20, 2, 6])  # [K, mcs_base, mcs_enh]
    print(f"\nBest params : {best}")
    print(f"Best fitness: {metrics['best_fitness']:.4f}")
    print(f"Time        : {metrics['duration_s']*1000:.1f} ms")

    print("\n--- Full GA vs BO comparison ---")
    compare_bo_vs_ga(snr_range=[5, 10, 15, 20, 25, 30], n_calls_bo=30)

    print("\n--- BO acquisition trace (for report figure) ---")
    plot_bo_acquisition(snr_db=20)
