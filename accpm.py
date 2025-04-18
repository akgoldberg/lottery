import numpy as np
import cvxpy as cp
from helpers import get_symmetric_intervals, partition_intervals
from cutting_plane import separation_oracle

def analytic_center(A: np.ndarray,
                                 b: np.ndarray,
                                 eps: float = 1e-9,
                                 tol: float = 1e-8):
    """
    Compute the analytic center of { x | A x < b } via CVXPY + MOSEK.
    Solves maximize sum(log(t)) subject to t == b - A x, t >= eps.
    Returns:
      x_opt (n,), t_opt (m,), nu (m,) dual multipliers for the equality constraints.
    """
    m, n = A.shape
    x = cp.Variable(n)
    t = cp.Variable(m)
    constraints = []
    # enforce t = b - A x
    constraints.append(t == b - A @ x)
    # maintain strict interior
    constraints.append(t >= eps)

    obj = cp.Maximize(cp.sum(cp.log(t)))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK,
               mosek_params={
                   "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
                   "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
                   "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
               })

    x_opt = x.value
    t_opt = t.value
    # duals for equality constraint t == b - A x is constraints[0]
    nu = constraints[0].dual_value
    return x_opt, t_opt, nu

def accpm_optimization(intervals, k,
                        use_symmetry=True,
                        add_monotonicity_constraints=True,
                        max_iters=100,
                        obj_tol=1e-8,
                        newton_tol=1e-6,
                        alpha=0.01,
                        beta=0.5,
                        max_center_iters=50,
                        tol=1e-6):
    """
    Analytic Center Cutting-Plane Method for max-min interval selection.
    Returns p_vals, v_val, and info dict.
    """
    # (A) symmetry / grouping
    if use_symmetry:
        sym_intervals = get_symmetric_intervals(intervals)
        n_vars = len(sym_intervals)
        i_to_var = {i: grp for grp, lst in enumerate(sym_intervals) for i in lst}
    else:
        n_vars = len(intervals)
        i_to_var = {i: i for i in range(n_vars)}

    # build initial constraint list in form a^T z <= b, z = [p (n_vars), v]
    # dimension d = n_vars + 1
    d = n_vars + 1
    constraints = []
    # sum p == k_pruned
    a = np.zeros(d); a[:n_vars] = 1
    constraints.append((a.copy(), float(k)))
    a = -a; constraints.append((a.copy(), float(-k)))
    # bounds 0 <= p_i <= 1
    for i in range(n_vars):
        a = np.zeros(d); a[i] = -1.0  # -p_i <= 0
        constraints.append((a.copy(), 0.0))
        a = np.zeros(d); a[i] = 1.0   # p_i <= 1
        constraints.append((a.copy(), 1.0))
    # initial top-k constraint: v <= sum_{j<k_pruned} p_j
    if k > 0:
        a = np.zeros(d)
        a[-1] = 1.0                     # v
        for j in range(min(k, len(intervals))):
            var = i_to_var[j]
            a[var] -= 1.0
        constraints.append((a.copy(), 0.0))

    # monotonicity constraints
    if add_monotonicity_constraints:
        _, partitions = partition_intervals(intervals, return_inds=True)
        for chain in partitions:
            for u, v_idx in zip(chain, chain[1:]):
                var_u, var_v = i_to_var[u], i_to_var[v_idx]
                if var_u != var_v:
                    a = np.zeros(d)
                    a[var_v] = 1.0  # p_v - p_u <= 0
                    a[var_u] = -1.0
                    constraints.append((a.copy(), 0.0))

    # ACCPM loop
    total_cuts = 0
    info = {'iterations': 0, 'convergence': False}

    best_lb = 0  # best feasible lower bound on v*
    best_ub = k  # best feasible upper bound on v*
    for it in range(max_iters):
        # (1) analytic center
        z, y, nu = analytic_center(constraints, tol=newton_tol,
                             alpha=alpha, beta=beta,
                             max_center_iters=max_center_iters)
        p_vars = z[:n_vars]
        p_vals = np.array([p_vars[i_to_var[i]] for i in range(len(intervals))])
        v_val = z[-1]

        # Compute dual-based upper bound from Section 4 of notes
        # slacks = y = b - A z, tau = 1/slacks
        # NOTE: could also solve full LP here with current cuts, but this approach should be more efficient
        b_vec = np.array([b for _, b in constraints])
        tau = 1.0 / y
        lam = tau / np.sum(tau)
        ub = lam.dot(b_vec)
        best_ub = min(best_ub, ub)

        # (2) separation oracle
        cuts = separation_oracle(p_vals, v_val, intervals, k, tol=tol)
        if len(cuts) == 0: # feasible 
            best_lb = max(best_lb, v_val)
            # check convergence
            if abs(best_lb - best_ub) < obj_tol: 
                info['convergence'] = True
                info['total_cuts'] = total_cuts
                info['iterations'] = it + 1
                return p_vals, v_val, info
            # add objective cut: v >= best_lb
            obj_cut = np.zeros(d)
            obj_cut[-1] = -1
            constraints.append((obj_cut.copy(), -best_lb))
            total_cuts += 1      
        else:   # infeasible  
        # (3) add new cuts
            for C in cuts:
                a = np.zeros(d)
                a[-1] = 1.0
                for idx in C:
                    var = i_to_var[idx]
                    a[var] -= 1.0
                constraints.append((a, 0.0))
            total_cuts += len(cuts)
        info['iterations'] = it + 1
    
    # no convergence
    info['total_cuts'] = total_cuts
    return p_vals, v_val, info
