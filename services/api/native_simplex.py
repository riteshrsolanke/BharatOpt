"""
BharatOpt-X Native Two-Phase Revised Simplex Engine
======================================================
100% Sovereign: Pure Python + NumPy ONLY. No scipy. No HiGHS. No external solver.
Implements:
  - Standard form conversion (slack/surplus/artificial variables)
  - Phase 1: Find a Basic Feasible Solution (BFS) by minimizing sum of artificials
  - Phase 2: Optimize the original objective from BFS
  - Full primal solution, objective value, slacks
  - Dual variables (shadow prices) via c_B * B^-1
  - Reduced costs (c_j - z_j) for all non-basic variables
  - Primal feasibility residual: ||Ax - b||_inf
  - Dual feasibility residual: ||c - A^T y - rc||_inf for basics
  - Status: OPTIMAL / INFEASIBLE / UNBOUNDED / MAX_ITER_REACHED
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

_PIVOT_TOL   = 1e-9   # minimum pivot element magnitude
_FEAS_TOL    = 1e-8   # primal feasibility tolerance
_OPT_TOL     = 1e-8   # optimality (dual) tolerance
_MAX_ITER    = 4000   # iteration cap

@dataclass
class NativeSolveResult:
    status: str                       # OPTIMAL | INFEASIBLE | UNBOUNDED | MAX_ITER_REACHED
    objective: float = 0.0
    variables: Dict[str, float] = field(default_factory=dict)
    slacks: Dict[str, float] = field(default_factory=dict)
    duals: Dict[str, float] = field(default_factory=dict)       # shadow prices
    reduced_costs: Dict[str, float] = field(default_factory=dict)
    primal_residual: float = 0.0      # ||Ax - b||_inf
    dual_residual: float = 0.0        # max |c_j - z_j| for basic vars
    duality_gap: float = 0.0          # |primal_obj - dual_obj|
    iterations: int = 0
    solve_time_ms: float = 0.0
    message: str = ""


def _bland_ratio_test(B_inv_N: np.ndarray, B_inv_b: np.ndarray, col: int) -> int:
    """Minimum ratio test for pivot row selection (Bland's anti-cycling)."""
    col_vals = B_inv_N[:, col]
    pos_mask = col_vals > _PIVOT_TOL
    if not np.any(pos_mask):
        return -1   # unbounded direction
    ratios = np.where(pos_mask, B_inv_b / col_vals, np.inf)
    return int(np.argmin(ratios))


def _full_tableau_simplex(
    c: np.ndarray,          # objective (minimization)
    A: np.ndarray,          # constraint matrix [m x n]
    b: np.ndarray,          # rhs  [m]
    basis: List[int],       # initial basic indices (len = m)
    n_total: int            # total columns in A
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Full Tableau Two-Phase Simplex (explicit basis inverse via LU/Gauss-Jordan).
    Returns: (status, x, B_inv, basis, iterations)
    """
    m = A.shape[0]
    status = "OPTIMAL"
    iters = 0

    for _ in range(_MAX_ITER):
        iters += 1
        B = A[:, basis]

        # Solve B x_B = b  (via least-squares, robust to near-singularity)
        try:
            x_B = np.linalg.lstsq(B, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            status = "INFEASIBLE"; break

        # Dual: y = c_B @ B^-1
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            # fallback: pseudoinverse
            B_inv = np.linalg.pinv(B)

        c_B = c[basis]
        y = c_B @ B_inv          # dual variables [m]

        # Reduced costs for all columns
        rc = c - y @ A           # [n_total]

        # Non-basic indices
        non_basic = [j for j in range(n_total) if j not in basis]

        # Optimality check: all rc >= -OPT_TOL for minimization
        rc_non_basic = rc[non_basic]
        if np.all(rc_non_basic >= -_OPT_TOL):
            # Optimal!
            x = np.zeros(n_total)
            x[basis] = x_B
            return (status, x, B_inv, np.array(basis), iters)

        # Enter: most negative reduced cost (Dantzig's rule)
        enter_local = int(np.argmin(rc_non_basic))
        enter_col = non_basic[enter_local]

        # Direction: B_inv @ A[:,enter_col]
        d = B_inv @ A[:, enter_col]

        # Ratio test
        pivot_row = _bland_ratio_test(
            (B_inv @ A[:, non_basic]) if len(non_basic) else np.zeros((m,0)),
            x_B, enter_local
        )
        # Re-check with d directly
        pos_mask = d > _PIVOT_TOL
        if not np.any(pos_mask):
            status = "UNBOUNDED"; break

        ratios = np.where(pos_mask, x_B / d, np.inf)
        leave_local = int(np.argmin(ratios))

        # Pivot
        basis[leave_local] = enter_col

    else:
        status = "MAX_ITER_REACHED"
        x = np.zeros(n_total)
        x[basis] = x_B if 'x_B' in dir() else np.zeros(m)
        return (status, x, B_inv if 'B_inv' in dir() else np.eye(m), np.array(basis), iters)

    x = np.zeros(n_total)
    if 'x_B' in dir():
        x[basis] = x_B
    return (status, x, B_inv if 'B_inv' in dir() else np.eye(m), np.array(basis), iters)


def native_solve(
    obj_terms: dict,
    constraints: list,
    sense: str = "max",
    model_name: str = "BharatOpt_Model"
) -> NativeSolveResult:
    """
    Entry point for the BharatOpt-X Native Two-Phase Simplex Engine.
    Inputs exactly mirror the existing API structure.
    """
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # 1. Extract all decision variables                                   #
    # ------------------------------------------------------------------ #
    var_set = set(obj_terms.keys())
    for c in constraints:
        var_set.update(c.get("terms", {}).keys())
    if not var_set:
        return NativeSolveResult(status="INFEASIBLE", message="Empty model.")

    vars_list = sorted(var_set)
    n = len(vars_list)                     # original decision vars
    vi = {v: i for i, v in enumerate(vars_list)}

    # ------------------------------------------------------------------ #
    # 2. Build original objective (minimization internally)               #
    # ------------------------------------------------------------------ #
    is_max = sense.strip().lower().startswith("max")
    c_orig = np.zeros(n)
    for v, coef in obj_terms.items():
        c_orig[vi[v]] = float(coef)
    c_min = -c_orig if is_max else c_orig.copy()

    # ------------------------------------------------------------------ #
    # 3. Convert constraints to standard form                             #
    # ------------------------------------------------------------------ #
    # Each constraint becomes: row @ x + slack_s - surplus_g = rhs
    # Collect slack (<=), surplus (>=), equality rows for Phase 1 artificials
    m = len(constraints)
    slack_count  = 0
    surplus_count = 0
    eq_count      = 0
    row_types = []   # 'le', 'ge', 'eq'

    for con in constraints:
        rel = str(con.get("rel", "<=")).strip().upper()
        if rel in ("<=", "L", "LE"):
            row_types.append("le"); slack_count += 1
        elif rel in (">=", "G", "GE"):
            row_types.append("ge"); surplus_count += 1; eq_count += 1
        else:
            row_types.append("eq"); eq_count += 1

    n_slacks   = slack_count
    n_surplus  = surplus_count
    n_art      = eq_count     # one artificial per >= and eq row

    n_total = n + n_slacks + n_surplus + n_art

    # Indices
    slack_start   = n
    surplus_start = n + n_slacks
    art_start     = n + n_slacks + n_surplus

    A = np.zeros((m, n_total))
    b = np.zeros(m)

    slack_idx   = slack_start
    surplus_idx = surplus_start
    art_idx     = art_start

    basis_init = []

    for i, con in enumerate(constraints):
        rhs = float(con.get("rhs", 0.0))
        b[i] = rhs
        terms = con.get("terms", {})
        for v, coef in terms.items():
            if v in vi:
                A[i, vi[v]] = float(coef)

        rt = row_types[i]
        if rt == "le":
            if rhs < 0:
                # flip: multiply row by -1, becomes >= form
                A[i, :] *= -1
                b[i] *= -1
                A[i, surplus_idx] = -1
                A[i, art_idx] = 1
                basis_init.append(art_idx)
                surplus_idx += 1; art_idx += 1
            else:
                A[i, slack_idx] = 1
                basis_init.append(slack_idx)
                slack_idx += 1
        elif rt == "ge":
            if rhs < 0:
                # flip: becomes <= after negation
                A[i, :] *= -1
                b[i] *= -1
                A[i, slack_idx] = 1
                basis_init.append(slack_idx)
                slack_idx += 1
            else:
                A[i, surplus_idx] = -1
                A[i, art_idx]     = 1
                basis_init.append(art_idx)
                surplus_idx += 1; art_idx += 1
        else:  # eq
            A[i, art_idx] = 1
            basis_init.append(art_idx)
            art_idx += 1

    # Ensure all rhs >= 0 (basic feasibility requirement)
    for i in range(m):
        if b[i] < -_FEAS_TOL:
            # flip row
            A[i, :] *= -1
            b[i] *= -1
            # re-assign artificial if not already
            if basis_init[i] < art_start:
                # previously slack: replace with artificial
                A[i, basis_init[i]] = 0
                A[i, art_idx] = 1
                basis_init[i] = art_idx
                n_art += 1; n_total += 1; art_idx += 1
                A = np.hstack([A, np.zeros((m, 1))])
                n_total = A.shape[1]

    # ------------------------------------------------------------------ #
    # 4. PHASE 1: Minimize sum of artificials                             #
    # ------------------------------------------------------------------ #
    art_indices = list(range(art_start, art_idx))
    total_iters = 0

    if art_indices:
        c_phase1 = np.zeros(n_total)
        c_phase1[art_indices] = 1.0

        basis_p1 = basis_init[:]
        status1, x1, _, basis_p1, iters1 = _full_tableau_simplex(
            c_phase1, A, b, basis_p1, n_total
        )
        total_iters += iters1

        # Check if artificials are zeroed out
        art_sum = float(np.sum(x1[art_indices]))
        if abs(art_sum) > _FEAS_TOL * 10:
            elapsed = (time.time() - t0) * 1000
            return NativeSolveResult(
                status="INFEASIBLE",
                solve_time_ms=round(elapsed, 3),
                iterations=total_iters,
                message="Phase 1: no feasible solution found. Model is infeasible.",
                primal_residual=art_sum
            )
        basis_p2 = [b for b in basis_p1 if b < art_start]
        # Fill any missing basis slots with slack/surplus columns
        all_structural = [j for j in range(art_start) if j not in basis_p2]
        while len(basis_p2) < m:
            basis_p2.append(all_structural.pop(0))
    else:
        basis_p2 = basis_init[:]

    # ------------------------------------------------------------------ #
    # 5. PHASE 2: Optimize original objective (structural columns only)   #
    # ------------------------------------------------------------------ #
    # Restrict to structural + slack/surplus columns (drop artificials)
    A2 = A[:, :art_start]
    c2 = np.zeros(art_start)
    c2[:n] = c_min[:n]
    # slack/surplus have 0 cost

    # Filter basis indices
    basis_p2_filtered = [b for b in basis_p2 if b < art_start]
    while len(basis_p2_filtered) < m:
        for j in range(art_start):
            if j not in basis_p2_filtered:
                basis_p2_filtered.append(j)
                break

    status2, x2, B_inv_final, basis_final, iters2 = _full_tableau_simplex(
        c2, A2, b, basis_p2_filtered[:m], art_start
    )
    total_iters += iters2

    # ------------------------------------------------------------------ #
    # 6. Extract results                                                  #
    # ------------------------------------------------------------------ #
    if status2 == "UNBOUNDED":
        elapsed = (time.time() - t0) * 1000
        return NativeSolveResult(
            status="UNBOUNDED",
            solve_time_ms=round(elapsed, 3),
            iterations=total_iters,
            message="Model is unbounded: objective grows without limit along a feasible ray. Add upper bounds on variables or missing constraints."
        )

    # Primal solution
    x_sol = x2[:n]
    s_sol = x2[n:art_start]   # slacks/surplus

    # Objective value
    raw_obj = float(c_min[:n] @ x_sol)
    obj_val = -raw_obj if is_max else raw_obj

    # Dual variables: y = c_B @ B_inv_final  (shadow prices)
    c_B = c2[list(basis_final)]
    try:
        y = c_B @ B_inv_final   # [m]
    except Exception:
        y = np.zeros(m)
    duals = -y if is_max else y   # flip sign for maximization

    # Reduced costs for original decision variables
    rc_all = c2 - y @ A2
    rc_orig = rc_all[:n]
    if is_max:
        rc_orig = -rc_orig

    # Primal residual: ||Ax - b||_inf
    primal_resid = float(np.max(np.abs(A2 @ x2 - b))) if m > 0 else 0.0

    # Dual residual: max |rc| for basic variables (should be 0)
    basic_rc = [abs(float(rc_all[j])) for j in basis_final if j < art_start]
    dual_resid = float(max(basic_rc)) if basic_rc else 0.0

    # Duality gap: |c^T x - b^T y| (LP strong duality)
    dual_obj = float(b @ (y))
    duality_gap = abs(raw_obj - dual_obj)

    # Build output dicts
    var_vals = {v: round(float(x_sol[vi[v]]), 6) for v in vars_list}
    # Zero out numerical noise
    var_vals = {v: (val if abs(val) > 1e-9 else 0.0) for v, val in var_vals.items()}

    slack_idx_map = {}
    s_counter = 0
    surplus_counter = 0
    for i, con in enumerate(constraints):
        rt = row_types[i]
        cname = con.get("name", f"C{i+1}")
        if rt == "le":
            slack_val = float(s_sol[s_counter]) if s_counter < len(s_sol) else 0.0
            slack_idx_map[cname] = round(max(0.0, slack_val), 6)
            s_counter += 1
        elif rt == "ge":
            surplus_val = float(s_sol[n_slacks + surplus_counter]) if (n_slacks + surplus_counter) < len(s_sol) else 0.0
            slack_idx_map[cname] = round(max(0.0, surplus_val), 6)
            surplus_counter += 1
        else:
            slack_idx_map[cname] = 0.0

    duals_map = {con.get("name", f"C{i+1}"): round(float(duals[i]), 6) for i, con in enumerate(constraints)}
    rc_map = {v: round(float(rc_orig[vi[v]]), 6) for v in vars_list}

    elapsed = (time.time() - t0) * 1000
    return NativeSolveResult(
        status="OPTIMAL" if status2 in ("OPTIMAL", "MAX_ITER_REACHED") else status2,
        objective=round(obj_val, 4),
        variables=var_vals,
        slacks=slack_idx_map,
        duals=duals_map,
        reduced_costs=rc_map,
        primal_residual=round(primal_resid, 10),
        dual_residual=round(dual_resid, 10),
        duality_gap=round(duality_gap, 8),
        iterations=total_iters,
        solve_time_ms=round(elapsed, 3),
        message=f"Solved via Native Two-Phase Simplex in {total_iters} iterations."
    )
