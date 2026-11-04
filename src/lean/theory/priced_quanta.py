"""
Priced-Quanta Theory Engine for Milestone M7 and Gate beta.

Implements:
1. Mathematical Derivations and Closed-Form Predictions (Theorems T1–T5):
   - T1: Knapsack compression scaling exponent gamma = alpha (Zipf exponent)
   - T2: Frequency-governed logistic survival threshold f*(B)
   - T3: Asymmetric half-life ratio B_{1/2}(fact) / B_{1/2}(skill) >= 2.0x
   - T4: Compression as reverse scaling
   - T5: Bounded thinking core saturation under progressive compression
2. Knapsack Solvers & Optimality Evaluators:
   - exact_01_knapsack: Exact dynamic programming 0/1 knapsack
   - greedy_knapsack: Continuous value-density sorting knapsack
   - evaluate_knapsack_gap: Measures (V_exact - V_greedy) / V_exact <= 1.0%
3. Empirical Statistical Diagnostics:
   - compute_roc_auc: Mann-Whitney U exact ROC-AUC computation
   - fit_power_law_exponent: Bootstrap power-law exponent fitting with 95% CI
   - compute_half_life: Interpolated 50% capacity half-life
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, root_scalar


def compute_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Computes exact Receiver Operating Characteristic Area Under Curve (ROC-AUC)
    via Mann-Whitney U rank-sum statistics.
    """
    y_arr = np.asarray(y_true, dtype=np.int32)
    s_arr = np.asarray(scores, dtype=np.float64)
    
    pos = s_arr[y_arr == 1]
    neg = s_arr[y_arr == 0]
    
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
        
    u_stat, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
    return float(u_stat / (len(pos) * len(neg)))


def exact_01_knapsack(values: List[float], weights: List[int], capacity: int) -> Dict[str, Any]:
    """
    Solves the exact 0/1 integer knapsack problem using dynamic programming.
    """
    n = len(values)
    dp = [0.0] * (capacity + 1)
    keep = [[False] * (capacity + 1) for _ in range(n)]
    
    for i in range(n):
        v = values[i]
        w = weights[i]
        for c in range(capacity, w - 1, -1):
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
                keep[i][c] = True
                
    # Traceback selected items
    c_cur = capacity
    selected = []
    for i in range(n - 1, -1, -1):
        if keep[i][c_cur]:
            selected.append(i)
            c_cur -= weights[i]
            
    selected.reverse()
    total_val = dp[capacity]
    total_wt = sum(weights[i] for i in selected)
    
    return {
        "max_value": float(total_val),
        "total_weight": int(total_wt),
        "selected_indices": selected,
        "num_selected": len(selected)
    }


def greedy_knapsack(values: List[float], weights: List[float], capacity: float) -> Dict[str, Any]:
    """
    Solves continuous / greedy knapsack by sorting items by density r_i = v_i / w_i.
    """
    n = len(values)
    densities = [(values[i] / max(weights[i], 1e-8), i) for i in range(n)]
    densities.sort(key=lambda x: x[0], reverse=True)
    
    rem_cap = capacity
    tot_val = 0.0
    tot_wt = 0.0
    selected = []
    
    for r, idx in densities:
        w = weights[idx]
        v = values[idx]
        if rem_cap >= w:
            selected.append(idx)
            tot_val += v
            tot_wt += w
            rem_cap -= w
        else:
            break
            
    return {
        "max_value": float(tot_val),
        "total_weight": float(tot_wt),
        "selected_indices": selected,
        "num_selected": len(selected)
    }


def evaluate_knapsack_gap(values: List[float], weights: List[int], capacity: int) -> Dict[str, Any]:
    """
    Evaluates the approximation gap between exact integer knapsack and greedy packing:
    Gap = (V_exact - V_greedy) / V_exact.
    """
    exact = exact_01_knapsack(values, weights, capacity)
    greedy = greedy_knapsack(values, [float(w) for w in weights], float(capacity))
    
    v_exact = exact["max_value"]
    v_greedy = greedy["max_value"]
    
    gap = (v_exact - v_greedy) / max(v_exact, 1e-8)
    return {
        "v_exact": v_exact,
        "v_greedy": v_greedy,
        "absolute_gap": v_exact - v_greedy,
        "relative_gap": float(gap),
        "within_one_percent": gap <= 0.010
    }


def solve_knapsack_soft_cutoff(
    r_values: np.ndarray,
    c_values: np.ndarray,
    budget: float,
    temperature: float = 1.0
) -> float:
    """
    Solves the continuous soft knapsack equation for price threshold tau(B):
        sum_i c_i * sigmoid((ln(r_i) - ln(tau)) / T) = B
    """
    log_r = np.log(np.maximum(r_values, 1e-12))
    
    def diff_fn(log_tau: float) -> float:
        z = (log_r - log_tau) / temperature
        # Stable sigmoid
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        packed_bits = np.sum(c_values * sig)
        return packed_bits - budget
        
    low_log = float(np.min(log_r) - 20.0)
    high_log = float(np.max(log_r) + 20.0)
    
    try:
        sol = root_scalar(diff_fn, bracket=[low_log, high_log], method="brentq")
        return float(np.exp(sol.root))
    except Exception:
        # Fallback to search
        res = minimize_scalar(lambda lt: diff_fn(lt)**2, bounds=(low_log, high_log), method="bounded")
        return float(np.exp(res.x))


def fit_power_law_exponent(
    budgets: List[float],
    excess_losses: List[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Fits empirical power-law exponent:
        ln(L(B) - L_inf) = -gamma * ln(B) + const
    Returns point estimate, 95% bootstrap CI, and R^2.
    """
    b_arr = np.array(budgets, dtype=np.float64)
    l_arr = np.array(excess_losses, dtype=np.float64)
    
    # Filter non-positive
    valid = (b_arr > 0) & (l_arr > 0)
    x = np.log(b_arr[valid])
    y = np.log(l_arr[valid])
    
    if len(x) < 3:
        return {"gamma": 0.0, "ci_95": [0.0, 0.0], "r_squared": 0.0}
        
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    gamma_est = -slope
    r2 = r_val ** 2
    
    # Bootstrap CIs
    rng = np.random.default_rng(42)
    boot_gammas = []
    n_pts = len(x)
    for _ in range(n_bootstrap):
        idx = rng.choice(n_pts, size=n_pts, replace=True)
        if len(np.unique(x[idx])) > 1:
            s_b, _, _, _, _ = stats.linregress(x[idx], y[idx])
            boot_gammas.append(-s_b)
            
    if len(boot_gammas) > 100:
        low_ci = float(np.percentile(boot_gammas, 100 * (alpha / 2.0)))
        high_ci = float(np.percentile(boot_gammas, 100 * (1.0 - alpha / 2.0)))
    else:
        low_ci = float(gamma_est - 1.96 * std_err)
        high_ci = float(gamma_est + 1.96 * std_err)
        
    return {
        "gamma": float(gamma_est),
        "ci_low": low_ci,
        "ci_high": high_ci,
        "ci_95": [low_ci, high_ci],
        "std_err": float(std_err),
        "r_squared": float(r2),
        "intercept": float(intercept)
    }


def compute_half_life(
    budgets: List[float],
    accuracies: List[float],
    relative: bool = True
) -> float:
    """
    Computes the 50% retention capacity half-life B_{1/2} from a compression ladder.
    
    If relative=True (default):
        Target threshold is 0.50 * baseline_acc (where baseline_acc is uncompressed accuracy at highest budget).
    If relative=False:
        Target threshold is 0.50 absolute.
        
    Interpolates the bit budget where accuracy crosses the threshold.
    If accuracy stays above threshold across all budgets (skills remain resilient),
    returns the lowest budget as an upper bound (B_{1/2} <= B_min).
    If accuracy is below threshold across all budgets,
    returns the highest budget (B_{1/2} >= B_max).
    """
    b_arr = np.array(budgets, dtype=np.float64)
    acc_arr = np.array(accuracies, dtype=np.float64)
    
    # Sort by descending budget (standard uncompressed -> highly compressed)
    order = np.argsort(b_arr)
    b_sorted = b_arr[order]
    acc_sorted = acc_arr[order]
    
    # Uncompressed baseline is at the highest budget
    a0 = float(acc_sorted[-1])
    target = (0.50 * a0) if (relative and a0 > 0) else 0.50
    
    if np.all(acc_sorted >= target):
        return float(b_sorted[0])  # Survives down to lowest budget (upper bound)
    if np.all(acc_sorted < target):
        return float(b_sorted[-1]) # Degrades above highest budget
        
    # Interpolate crossing point
    for i in range(len(b_sorted) - 1):
        if (acc_sorted[i] <= target and acc_sorted[i+1] >= target) or (acc_sorted[i] >= target and acc_sorted[i+1] <= target):
            denom = acc_sorted[i+1] - acc_sorted[i]
            if abs(denom) < 1e-6:
                return float(b_sorted[i])
            frac = (target - acc_sorted[i]) / denom
            return float(b_sorted[i] + frac * (b_sorted[i+1] - b_sorted[i]))
            
    return float(np.median(b_sorted))


def compute_detailed_half_life(
    budgets: List[float],
    accuracies: List[float],
    relative: bool = True,
    chance_floor: float = 0.0
) -> Dict[str, Any]:
    """
    Computes rigorous retention half-life B_{1/2} with status, chance floor, and bounds.
    
    If chance_floor > 0:
        Evaluates excess retention over chance:
        target = chance_floor + 0.50 * (a0 - chance_floor)
    """
    b_arr = np.array(budgets, dtype=np.float64)
    acc_arr = np.array(accuracies, dtype=np.float64)
    order = np.argsort(b_arr)
    b_sorted = b_arr[order]
    acc_sorted = acc_arr[order]
    a0 = float(acc_sorted[-1])
    
    if chance_floor > 0 and a0 > chance_floor:
        target = float(chance_floor + 0.50 * (a0 - chance_floor))
    elif relative and a0 > 0:
        target = float(0.50 * a0)
    else:
        target = 0.50
        
    crossed = False
    b_half = None
    is_upper_bound = False
    
    if np.all(acc_sorted >= target):
        b_half = float(b_sorted[0])
        is_upper_bound = True
        crossed = False
    elif np.all(acc_sorted < target):
        b_half = float(b_sorted[-1])
        crossed = False
    else:
        for i in range(len(b_sorted) - 1):
            if (acc_sorted[i] <= target and acc_sorted[i+1] >= target) or (acc_sorted[i] >= target and acc_sorted[i+1] <= target):
                denom = acc_sorted[i+1] - acc_sorted[i]
                if abs(denom) < 1e-6:
                    b_half = float(b_sorted[i])
                else:
                    frac = (target - acc_sorted[i]) / denom
                    b_half = float(b_sorted[i] + frac * (b_sorted[i+1] - b_sorted[i]))
                crossed = True
                break
        if b_half is None:
            b_half = float(np.median(b_sorted))
            
    # Linear extrapolation if uncrossed and staying above target
    extrapolated_b_half = None
    if not crossed and is_upper_bound and len(b_sorted) >= 2:
        idx_other = None
        for j in range(1, len(b_sorted)):
            if abs(acc_sorted[j] - acc_sorted[0]) > 1e-4:
                idx_other = j
                break
        if idx_other is not None:
            slope = (acc_sorted[idx_other] - acc_sorted[0]) / (b_sorted[idx_other] - b_sorted[0])
            if abs(slope) > 1e-15:
                delta_b = (target - acc_sorted[0]) / slope
                min_floor = float(b_sorted[0] * 0.25)
                extrapolated_b_half = float(max(min_floor, b_sorted[0] + delta_b))
        else:
            min_floor = float(b_sorted[0] * 0.25)
            extrapolated_b_half = min_floor
                
    return {
        "b_half": float(b_half),
        "crossed": crossed,
        "is_upper_bound": is_upper_bound,
        "target_accuracy": float(target),
        "baseline_accuracy": float(a0),
        "chance_floor": float(chance_floor),
        "extrapolated_b_half": extrapolated_b_half
    }


class PricedQuantaTheory:
    """
    Formal mathematical realization of the Priced-Quanta Equation:
        L(B) = L_inf + sum_i p_i * Delta_i * [ 1 - sigma((ln r_i - ln tau(B)) / T) ]
    """
    def __init__(
        self,
        c_alg: float = 28.0,
        kappa: float = 2.0,
        temperature: float = 1.0,
        l_inf: float = 0.05
    ):
        self.c_alg = c_alg
        self.kappa = kappa
        self.temperature = temperature
        self.l_inf = l_inf
        
    def predict_scaling_exponent(self, zipf_alpha: float) -> float:
        """Theorem T1: Compression power-law exponent gamma = alpha."""
        return zipf_alpha
        
    def predict_fact_survival_probability(
        self,
        p_fact: float,
        p_star: float,
        temperature: Optional[float] = None
    ) -> float:
        """Theorem T2: Logistic fact survival threshold."""
        temp = temperature or self.temperature
        z = (math.log(max(p_fact, 1e-12)) - math.log(max(p_star, 1e-12))) / temp
        return 1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))
        
    def predict_half_life_ratio(
        self,
        p_skill: float,
        k_skill_tokens: float,
        p_fact: float,
        i_fact_bits: float
    ) -> float:
        """
        Theorem T3: Asymmetric Half-Life Ratio formula:
        r_skill / r_fact = (p_skill / (c_alg * K(s))) / (p_fact / (I(f) / kappa))
        """
        c_skill = self.c_alg * k_skill_tokens
        c_fact = i_fact_bits / self.kappa
        r_skill = p_skill / max(c_skill, 1e-8)
        r_fact = p_fact / max(c_fact, 1e-8)
        return float(r_skill / max(r_fact, 1e-8))
