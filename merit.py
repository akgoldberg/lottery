import gurobipy as gp
from gurobipy import GRB
import numpy as np
from helpers import prune_instance, partition_intervals, get_symmetric_intervals, verify_monotonicity_in_k
import time

#######################################################################
#                          Solve Optimization Problem                 #
#######################################################################

# Separation oracle for the cutting plane method
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

# Cutting plane optimization method
def cutting_plane_optimization(intervals, k, p_lower_bound,
                                use_symmetry, add_monotonicity_constraints,
                                max_iters, max_cuts_per_iter, tol, drop_cut_limit,
                                verbose, print_iter):   
    # Check that intervals given are sorted by left endpoint
    if not all(intervals[i][0] >= intervals[i+1][0] for i in range(len(intervals)-1)):
        unsorted = [(intervals[i][0], intervals[i+1][0]) for i in range(len(intervals)-1) if intervals[i][0] < intervals[i+1][0]]
        s = f"Intervals not sorted by left endpoint. Unsorted intervals: {unsorted}"
        assert False, s

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

###############################################################################
#             Ex Post Validity Post-Processing                                #
###############################################################################

# Ensure that if i dominates j, then either p_i = 1 or p_j = 0
def postprocess_solution(p, intervals):
    items = list(zip(p, intervals, range(len(p))))
    
    # Sort items by ascending order of p
    items.sort(key=lambda x: x[0])
    
    # Extract the sorted probabilities, intervals, and original indices
    sorted_p, sorted_intervals, original_indices = zip(*items)
    
    # Convert sorted_p to a list for mutability
    sorted_p = list(sorted_p)
    
    n = len(sorted_p)
    
    # Iterate over each element in the sorted list
    for b in range(n):
        if sorted_p[b] == 0:
            continue
        for a in range(n-1, b, -1):
            if sorted_intervals[a][0] > sorted_intervals[b][1] and sorted_p[a] < 1:
                d = min(sorted_p[b], 1 - sorted_p[a])
                sorted_p[b] -= d
                sorted_p[a] += d
    
    # Reorder the probabilities to match the original order
    adjusted_p = [0] * n
    for i, index in enumerate(original_indices):
        adjusted_p[index] = sorted_p[i]
    
    return adjusted_p

#######################################################################
#                           Sampling                                  #
#######################################################################

# Sample exaclty k items from a population with marginal probabilities given by p (p sum to k).
# Returns a list of indices of selected items
# Source: https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-20/issue-3/On-the-Theory-of-Systematic-Sampling-II/10.1214/aoms/1177729988.full
def systematic_sampling(k, p):
    n = len(p)
    assert np.isclose(sum(p), k), "Marginal probabilities must sum to k"

    # Randomly permute order of items
    perm = np.random.permutation(n)
    p = [p[i] for i in perm]

    # Compute cumulative probabilities with S[0] = 0
    S = np.cumsum(p)
    S = np.insert(S, 0, 0)  # Now length n+1
    
    # Generate sorted sampling points 
    u = np.random.uniform(0, 1)
    sampling_points = [u + m for m in range(k)]
    
    # Select items with each point in [S[j], S[j+1])
    selected = []
    j = 0  # Pointer to current interval
    for point in sampling_points:
        # Advance pointer until we find S[j] > point
        while j < len(S) and S[j] <= point:
            j += 1
        selected.append(perm[j-1])  # Items are 1-indexed, so we subtract 1
    
    return selected

# verify that systematic sampling works as expected 
def verify_sampling(n, k, p, num_trials=10_000):
    counts = np.zeros(n)
    
    for _ in range(num_trials):
        sample = systematic_sampling(k, p)
        for item in sample:
            counts[item] += 1
            
    empirical_p = counts / num_trials  # Correct normalization
    return empirical_p

#######################################################################
##                              Full Algorithm                       ##
#######################################################################

def run_merit(intervals, k, enforce_monotonicity=False):
    """
    Run the MERIT algorithm on the given intervals and k.
    
    Parameters:
    - intervals: List of tuples (low, high) representing the intervals.
    - k: Number of items to select.
    - enforce_monotonicity: If True, enforce monotonicity in budget (may lead to loss of optimality).
    
    Returns:
    - p: List of selection probabilities for each item.
    - selected_items: List of indices of selected items.
    """
    
    if enforce_monotonicity:
       p_seq, _, _ = solve_with_monotonicity(intervals, k)
       p = p_seq[-1]  # Last sequence corresponds to k
    else:
        p, v, info = solve_problem(intervals, k)
    
    # Post-process the solution to ensure ex post validity
    p = postprocess_solution(p, intervals)

    selected_items = systematic_sampling(k, p)
    return p, selected_items