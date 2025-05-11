import gurobipy as gp
from gurobipy import GRB
import numpy as np
from helpers import prune_instance, partition_intervals, get_symmetric_intervals, verify_monotonicity_in_k
import time

def separation_oracle(p_vals, v_val, intervals, k, tol=1e-6, max_cuts_per_iter=None):
    C_w = [] 
    w = []
    for i in range(k+1):
        # find all intervals that overlap with interval i
        S_i = [(j, p_vals[j]) for j in range(i+1, len(intervals)) if intervals[j][1] >= intervals[i][0]]
        if len(S_i) >= k - i:
            # sort S_i by p_j and take the k-i intervals with smallest p_j
            J_i = sorted(S_i, key=lambda x: x[1])[:k-i]
            v_i =  sum(p_vals[:i]) + sum(p_j for _, p_j in J_i)
            if v_i < v_val - tol:
                violated = list(range(i)) + [j for j, _ in J_i]
                C_w += [violated]
                w += [v_i]
    if max_cuts_per_iter is None:
        return C_w
    # return the top n_cuts constraints by smallest v_i
    C_w = sorted(zip(C_w, w), key=lambda x: x[1])[:max_cuts_per_iter]
    cuts = [C for C, _ in C_w]
    return cuts

def cutting_plane_optimization(intervals, k, p_lower_bound,
                                use_symmetry, add_monotonicity_constraints,
                                max_iters, max_cuts_per_iter, tol, drop_cut_limit,
                                verbose, print_iter):   
    # Check that intervals given are sorted by left endpoint
    assert all(intervals[i][0] >= intervals[i+1][0] for i in range(len(intervals)-1)), "Intervals must be sorted by left endpoint."

    timing_info = {}

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model(env=env)

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
    m.addConstr(gp.quicksum(p[i_to_var[i]] for i in range(len(intervals))) == k, name="sum_p")
    m.addConstrs((p[i_to_var[i]] >= p_lower_bound[i] for i in range(len(intervals))), name="p_LB")
    m.addConstrs((p[i] <= 1 for i in range(n_vars)), name="p_prob1")
    m.addConstr(v <= gp.quicksum(p[i_to_var[j]] for j in range(k)), name="topk_constraint")

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
    step_start = time.time()
    for iter_num in range(max_iters):
        m.optimize()

        if (drop_cut_limit is not None) and (current_cuts > drop_cut_limit*n_vars):
            # only keep cut constraints with drop_cut_limit * n_vars smallest slack values
            cut_slacks = [c.Slack for c in m.getConstrs() if c.ConstrName == "cut"]
            cutoff = sorted(cut_slacks)[drop_cut_limit*n_vars]
            for c in m.getConstrs():
                if c.ConstrName == "cut" and c.Slack > cutoff:
                    m.remove(c)
                    current_cuts -= 1
                    
        if m.status != GRB.OPTIMAL:
            raise ValueError("Problem is infeasible or unbounded")

        step_start = time.time()
        p_vars = [p[i].X for i in range(n_vars)]
        p_vals = [p_vars[i_to_var[i]] for i in range(len(intervals))]
        v_val = v.X

        # Separation oracle
        cuts = separation_oracle(p_vals, v_val, intervals, k, tol, max_cuts_per_iter)

        if len(cuts) == 0:
            timing_info['optimization_loop_time'] = time.time() - step_start
            if verbose:
                print(f"Iteration {iter_num}: Added {len(cuts)} constraints, total cuts: {total_cuts}, current_cuts: {current_cuts}, v_UB= {v_val:.4f}.")
            return p_vals, v_val, {'iterations': iter_num + 1,
                                    'convergence': True,
                                    'total_cuts': total_cuts,
                                    'n_vars': n_vars,
                                    'n_chains': n_chains,
                                    'n_mono_constraints': n_mono_constraints,
                                    'timing': timing_info}
        
        for C in cuts:
            m.addConstr(v <= gp.quicksum(p[i_to_var[i]] for i in C), name="cut")
        total_cuts += len(cuts)
        current_cuts += len(cuts)

        if verbose and iter_num % print_iter == 0:
            print(f"Iteration {iter_num}: Added {len(cuts)} constraints, total cuts: {total_cuts}, current_cuts: {current_cuts}, v_UB= {v_val:.4f}.")

    if verbose:
        print(f"Max iterations reached ({max_iters}) without convergence.")
    
    timing_info['optimization_loop_time'] = time.time() - step_start

    return p_vals, v_val, {'iterations': max_iters,
                            'convergence': False,
                            'total_cuts': total_cuts,
                            'n_vars': n_vars,
                            'n_chains': n_chains,
                            'n_mono_constraints': n_mono_constraints,
                            'timing': timing_info}

def solve_problem(intervals, k, set_p_lower_bound=None, sort_by_left=True,
                    init_prune=True, use_symmetry=True, add_monotonicity_constraints=True, 
                    max_iters=1000, tol=1e-6, max_cuts_per_iter=None, drop_cut_limit=3, print_iter=10, verbose=False):
    
    assert(k < len(intervals)), "k must be less than the number of intervals."
    assert(k >= 0), "k must be greater than or equal to 0."
    
    start_time = time.time()
    timing_info = {}

    if set_p_lower_bound is None:
        p_lower_bound = [0]*len(intervals)
    else:
        p_lower_bound = set_p_lower_bound
        init_prune = False # Do not prune if solving monotonic sequence

    if sort_by_left:
        order = np.argsort([-x[0] for x in intervals])
        intervals = [intervals[i] for i in order]
        p_lower_bound = [p_lower_bound[i] for i in order]

    step_time = time.time()
    # (1) prune all intervals that are always in the top k or never in the top k
    if init_prune: 
        indices_pruned, top, _ = prune_instance(intervals, k)
        intervals_pruned = [intervals[i] for i in indices_pruned]
        k_pruned = k - len(top)
        p_lower_bound_pruned = [p_lower_bound[i] for i in indices_pruned]

        if verbose:
            print(f'Pruned {len(intervals)-len(intervals_pruned)} intervals. Solving with n={len(intervals_pruned)}, k={k_pruned}.')
    else:
        indices_pruned, top = list(range(len(intervals))), []
        intervals_pruned = intervals
        k_pruned = k
        p_lower_bound_pruned = p_lower_bound
        if verbose:
            print(f'Not pruning the LP as a first step.')
    if k_pruned <= 0:
        info = {'iterations': 0,
                'convergence': True,
                'total_cuts': 0,
                'n_vars': len(intervals),
                'n_chains': None,
                'n_mono_constraints': 0,
                'timing': timing_info}
        p_out = [1]*k + [0]*(len(intervals)-k)
        v_out = k

        if sort_by_left:
            p_out = [p_out[i] for i in np.argsort(order)]

        return np.array(p_out), v_out, info
    timing_info['init_prune_time'] = time.time() - step_time

    # (2) optimize using cutting plane method
    p_vals, v_val, info = cutting_plane_optimization(intervals_pruned, k_pruned, p_lower_bound_pruned,
                                                        use_symmetry, add_monotonicity_constraints,
                                                        max_iters, max_cuts_per_iter, tol, drop_cut_limit,
                                                        verbose, print_iter)
    
    # (3) add pruned top and bottom intervals back to the solution with p=1 and p=0 respectively
    v_out = v_val + len(top)
    p_out = np.zeros(len(intervals))
    for i, p_i in enumerate(p_vals):
        p_out[indices_pruned[i]] = p_i
    p_out[top] = 1.
    p_out = np.clip(p_out, 0, 1)

    # add timing to the info dict
    info['timing']['total_time'] = time.time() - start_time
    info['timing']['init_prune_time'] = timing_info['init_prune_time']

    # re-order p_out to original order
    if sort_by_left:
        p_out = [p_out[i] for i in np.argsort(order)]
        p_out = np.array(p_out)

    return p_out, v_out, info

def solve_with_monotonicity(intervals, k, max_iters=1000, max_cuts_per_iter=None, drop_cut_limit=3,
                                print_iter=10, verbose=False, check_monotonicity=True):
    
    p_seq = []
    v_seq = []
    info_seq = []
    
    p_lower_bound = [0]*len(intervals)
    for i in range(1,k+1):
        print('Solving with n_selected =', i)  
        # solve with n_selected = i
        p,v,info  = solve_problem(intervals, i, set_p_lower_bound=p_lower_bound,
                                    sort_by_left=True, init_prune=False, use_symmetry=True,
                                    add_monotonicity_constraints=True,
                                    max_iters=max_iters, tol=1e-6, max_cuts_per_iter=max_cuts_per_iter,
                                    drop_cut_limit=drop_cut_limit, print_iter=print_iter, verbose=verbose)
        p_lower_bound = p
        p_seq.append(p)
        v_seq.append(v)
        info_seq.append(info)
    
    if check_monotonicity:
        verify_monotonicity_in_k(p_seq, raise_error=True)
    
    return p_seq, v_seq, info_seq