import numpy as np
import cvxpy as cp
from helpers import get_symmetric_intervals, partition_intervals, prune_instance
from cutting_plane import separation_oracle
from full_LP import solve_instance_ordered
import time

import gurobipy as gp
from gurobipy import GRB


def analytic_center(A, b, eps=1e-8, reg=0):
    _, n = A.shape
    x = cp.Variable(n)

    # Define the slack variables: s_i = b_i - a_i^T x > 0
    slack = b - A @ x

    # Objective: maximize sum(log(slack)) + regularization
    objective = cp.Maximize(cp.sum(cp.log(slack)) - reg * cp.sum_squares(x)/n)

    # Constraint: slack > 0 ⇔ Ax < b (ensure point is in interior)
    constraints = [slack >= eps]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, 
                  warm_start=True,
                  mosek_params={'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-9,
                                'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-9})

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Analytic center optimization failed: {problem.status}")

    return x.value, slack.value

def acppm_optimization(intervals, k,
                                use_symmetry, add_monotonicity_constraints,
                                max_iters, max_cuts_per_iter, oracle_tol, obj_tol, obj_bound_method, drop_cut_limit, solve_initial_LB, solve_LP_iter,
                                verbose, print_iter):   
    # Check that intervals given are sorted by left endpoint
    assert all(intervals[i][0] >= intervals[i+1][0] for i in range(len(intervals)-1)), "Intervals must be sorted by left endpoint."

    timing_info = {}

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model(env=env)

    step_start = time.time()
    # (1) Set initial lower bound
    if solve_initial_LB:
        # get ordering of intervals by left endpoint
        tau = sorted(range(len(intervals)), key=lambda i: -intervals[i][0])
        v_LB,_ = solve_instance_ordered(intervals, k, k, tau, verbose=False, prune=True, postprocess=True)
        if verbose:
            print(f"Setting initial lower bound from solving ordered problem: {v_LB:.4f}.")
    else:
        v_LB = 0
        if verbose:
            print(f"Setting initial lower bound to 0.")
    timing_info['initial_LB'] = time.time() - step_start

    # (A) Group all intervals with identical constraints to reduce number of decision vars
    step_start = time.time()
    if use_symmetry:
        sym_intervals = get_symmetric_intervals(intervals)
        n_vars = len(sym_intervals)
        i_to_var = {interval: group for group, lst in enumerate(sym_intervals) for interval in lst}

        if verbose:
            print(f"Using symmetry breaking. Number of decision vars: {n_vars}.")
    else:
        n_vars = len(intervals)
        i_to_var = {i: i for i in range(n_vars)}

        if verbose:
            print(f"Not using symmetry breaking. Number of decision vars: {n_vars}.")
    timing_info['symmetry_setup'] = time.time() - step_start

    v = m.addVar(vtype=GRB.CONTINUOUS, name="v")
    p = m.addVars(n_vars, vtype=GRB.CONTINUOUS, name="p")

    m.setObjective(v, GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(p[i_to_var[i]] for i in range(len(intervals))) <= k, name="sum_p")
    m.addConstrs((p[i] >= 0 for i in range(n_vars)), name="p_prob0")
    m.addConstrs((p[i] <= 1 for i in range(n_vars)), name="p_prob1")
    m.addConstr(v <= k, name="v_UB0")
    m.addConstr(v >= v_LB, name="v_LB")

    # (B) Add monotonicity constraints
    step_start = time.time()
    n_mono_constraints = 0
    n_chains = None 
    if add_monotonicity_constraints:
        _, partitions = partition_intervals(intervals, return_inds=True)
        n_chains = len(partitions)
        for part in partitions:
            for i in range(len(part)-1):
                if i_to_var[part[i]] != i_to_var[part[i+1]]:
                    m.addConstr(p[i_to_var[part[i]]] >= p[i_to_var[part[i+1]]], name="monotonicity")
                    n_mono_constraints += 1
        if verbose:
            print(f'Added initial monotonicity constraints: {n_mono_constraints} from {n_chains} chains.')
    timing_info['monotonicity_constraints_setup'] = time.time() - step_start

    total_cuts = 0
    current_cuts = 0
    feasibility_cuts = 0
    objective_cuts = 0
    step_start = time.time()

    v_UB = k
    best_feasible = None

    timing_info['analytic_center'] = 0
    timing_info['drop_cut_limit'] = 0
    timing_info['separation_oracle'] = 0
    timing_info['LB_update'] = 0
    timing_info['UB_update'] = 0

    for iter_num in range(max_iters):
        iter_start_time = time.time()

        # Solve the analytic center problem
        step_start = time.time()
        A, b, v_LB_idx = extract_constraints(m)
        x, slack = analytic_center(A, b, eps=1e-8)
        timing_info['analytic_center'] += time.time() - step_start

        p_vars = x[1:]  # Exclude the first variable (v)
        p_vals = [p_vars[i_to_var[i]] for i in range(len(intervals))]
        v_val = x[0]  # The first variable is v

        # Drop cuts if necessary
        step_start = time.time()
        if drop_cut_limit is not None:
            max_total_cuts = drop_cut_limit * n_vars
            if current_cuts > max_total_cuts:
                w = 1.0 / (slack**2)
                H = A.T.dot(A * w[:, None])
                H_inv = np.linalg.inv(H)
                HinvA_T = H_inv.dot(A.T)
                quad = np.einsum('ij,ij->i', A, HinvA_T.T)
                eta = slack / np.sqrt(quad)

                etas = [eta[i] for i, c in enumerate(m.getConstrs()) if c.ConstrName == "cut"]
                cutoff = sorted(etas)[max_total_cuts]
                for i, c in enumerate(m.getConstrs()):
                    if c.ConstrName == "cut" and etas[i] < cutoff:
                        m.remove(c)
                        current_cuts -= 1
        timing_info['drop_cut_limit'] += time.time() - step_start

        # Separation oracle
        step_start = time.time()
        cuts = separation_oracle(p_vals, v_val, intervals, k, oracle_tol, max_cuts_per_iter)
        timing_info['separation_oracle'] += time.time() - step_start

        if len(cuts) == 0:
            # If feasible solution found, update best feasible and add obj cut
            step_start = time.time()
            if v_val < v_LB:
                raise ValueError(f"No convergence: v_LB = {v_LB} > v_val = {v_val}.")
            v_LB = v_val
            best_feasible = p_vals, v_val
            m.getConstrByName("v_LB").RHS = v_LB
            total_cuts += 1
            objective_cuts += 1
            timing_info['LB_update'] += time.time() - step_start

        for C in cuts:
            m.addConstr(v <= gp.quicksum(p[i_to_var[i]] for i in C), name="cut")

        total_cuts += len(cuts)
        current_cuts += len(cuts)
        feasibility_cuts += len(cuts)

        # Get upper bound estimate
        step_start = time.time()
        if obj_bound_method == 'solve_LP':
            if len(cuts) > 0:
                m.optimize()
                v_UB = v.X
                # check feasibility
                p_vars = [p[i].X for i in range(n_vars)]
                p_vals = [p_vars[i_to_var[i]] for i in range(len(intervals))]
                check = separation_oracle(p_vals, v_UB, intervals, k, oracle_tol, max_cuts_per_iter)
                if len(check) == 0: # feasible so LB = UB = v_UB
                    v_LB = v_UB
                    best_feasible = p_vals, v_UB
        elif obj_bound_method == 'slack_vars':
            slack_obj = slack[v_LB_idx]
            slack = np.delete(slack, v_LB_idx)
            mu = slack_obj / slack
            b_mult = np.delete(b, v_LB_idx)
            v_UB_new = b_mult.dot(mu) + obj_tol
            if v_UB_new < v_UB:
                v_UB = v_UB_new
            if solve_LP_iter is not None and iter_num > 0 and iter_num % solve_LP_iter == 0:
                m.optimize()
                v_UB = min(v_UB, v.X)
                # check feasibility
                p_vars = [p[i].X for i in range(n_vars)]
                p_vals = [p_vars[i_to_var[i]] for i in range(len(intervals))]
                check = separation_oracle(p_vals, v_UB, intervals, k, oracle_tol, max_cuts_per_iter)
                if len(check) == 0: # feasible so LB = UB = v_UB
                    v_LB = v_UB
                    best_feasible = p_vals, v_UB
        else:
            raise ValueError(f"Unknown method for obtaining upper bound: {obj_bound_method}.")
        timing_info['UB_update'] += time.time() - step_start

        # Check for convergence
        if v_UB - v_LB < obj_tol:
            timing_info['optimization_loop_time'] = time.time() - step_start
            if verbose:
                print(f"Iteration {iter_num}: Added {len(cuts)} constraints, total cuts: {total_cuts}, active cuts: {current_cuts}, v_UB= {v_UB:.4f}, v_LB={v_LB:.4f}.")
            p_vals, v_val = best_feasible
            return p_vals, v_val, {'iterations': iter_num + 1,
                                    'convergence': True,
                                    'total_cuts': total_cuts,
                                    'feasibility_cuts': feasibility_cuts,
                                    'objective_cuts': objective_cuts,
                                    'n_vars': n_vars,
                                    'n_chains': n_chains,
                                    'n_mono_constraints': n_mono_constraints,
                                    'timing': timing_info}

        if verbose and iter_num % print_iter == 0:
            print(f"Iteration {iter_num}: Added {len(cuts)} constraints, total cuts: {total_cuts}, active cuts: {current_cuts}, v_UB= {v_UB:.4f}, v_LB={v_LB:.4f}.")
    if verbose:
        print(f"Max iterations reached ({max_iters}) without convergence.")
    
    timing_info['optimization_loop_time'] = time.time() - step_start

    return p_vals, v_val, {'iterations': max_iters,
                            'convergence': False,
                            'total_cuts': total_cuts,
                            'feasibility_cuts': feasibility_cuts,
                            'objective_cuts': objective_cuts,
                            'n_vars': n_vars,
                            'n_chains': n_chains,
                            'n_mono_constraints': n_mono_constraints,
                            'timing': timing_info}

def solve_problem(intervals, k, sort_by_left=True,
                    init_prune=True, use_symmetry=True, add_monotonicity_constraints=True, upper_bound_method='slack_vars',
                    max_iters=1000, oracle_tol=1e-6, obj_tol=1e-5, max_cuts_per_iter=None, drop_cut_limit=None, solve_initial_LB=False, solve_LP_iter=None,
                    print_iter=10, verbose=False):
    
    start_time = time.time()
    timing_info = {}

    if sort_by_left:
        intervals = sorted(intervals, key=lambda x: x[0], reverse=True)
    
    step_time = time.time()
    # (1) prune all intervals that are always in the top k or never in the top k
    if init_prune: 
        indices_pruned, top, bottom = prune_instance(intervals, k)
        intervals_pruned = [intervals[i] for i in indices_pruned]
        k_pruned = k - len(top)

        if verbose:
            print(f'Pruned {len(intervals)-len(intervals_pruned)} intervals. Solving with n={len(intervals_pruned)}, k={k_pruned}.')
    else:
        indices_pruned, top = list(range(len(intervals))), []
        intervals_pruned = intervals
        k_pruned = k
        if verbose:
            print(f'Not pruning the LP as a first step.')
    if k_pruned <= 0:
        return k, [1]*k + [0]*(len(intervals)-k)
    timing_info['init_prune_time'] = time.time() - step_time

    # (2) optimize using cutting plane method
    p_vals, v_val, info = acppm_optimization(intervals_pruned, k_pruned,
                                                        use_symmetry, add_monotonicity_constraints,
                                                        max_iters, max_cuts_per_iter, oracle_tol, obj_tol, upper_bound_method, drop_cut_limit, solve_initial_LB, solve_LP_iter,
                                                        verbose, print_iter)

    # (3) add pruned top and bottom intervals back to the solution with p=1 and p=0 respectively
    v_out = v_val + len(top)
    p_out = np.zeros(len(intervals))
    for i, p_i in enumerate(p_vals):
        p_out[indices_pruned[i]] = p_i
    p_out[top] = 1.
    p_out = np.clip(p_out, 0, 1)

    # add meta data on pruning to the info dict
    info['n_top'] = len(top)
    info['n_bottom'] = len(bottom)
    info['n_rand'] = len(intervals_pruned)

    # add timing to the info dict
    info['timing']['total_time'] = time.time() - start_time
    info['timing']['init_prune_time'] = timing_info['init_prune_time']

    return p_out, v_out, info

def extract_constraints(m):
    """Extract constraints from Gurobi as matrics A and b
       and the index of constraint v <= v_LB."""
    vars_list = m.getVars()
    var_names = [v.VarName for v in vars_list]
            
    # Get explicit constraints
    A_matrix = m.getA().toarray()
    b_vector = np.array(m.getAttr("RHS"))
    sense_vector = np.array(m.getAttr("Sense"))
    
    # Convert all constraints to the form Ax <= b
    for i, sense in enumerate(sense_vector):
        if sense == '>':
            A_matrix[i, :] = -A_matrix[i, :]
            b_vector[i] = -b_vector[i]
        elif sense == '=':
            # For equality constraints, add another constraint with opposite sign
            A_matrix = np.vstack([A_matrix, -A_matrix[i, :]])
            b_vector = np.append(b_vector, -b_vector[i])

    # Find the index of constraint "v_LB" in A
    # Find the index of the "v_LB" constraint in the model's constraints
    for i, c in enumerate(m.getConstrs()):
        if c.ConstrName == "v_LB":
            v_LB_idx = i
            break

    return A_matrix, b_vector, v_LB_idx