import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

from helpers import prune_instance, sort_intervals, partition_intervals, postprocess_solution, evaluate_p

def solve_instance_base(intervals, T, k, add_adversary_constraints, lex_order_p=0, verbose=True, prune=True, p_lower_bound=None, postprocess=True):
    if verbose:
        print(f'Solving instance with n={len(intervals)}, k={k}.')

    if prune: 
        # (1) prune all intervals that are always in the top T or never in the top T
        indices_pruned, top, _ = prune_instance(intervals, T)
        intervals_pruned = [intervals[i] for i in indices_pruned]
        k_pruned = k - len(top)
        T_pruned = T - len(top)

        if verbose:
            print(f'Pruned {len(intervals)-len(intervals_pruned)} intervals. Solving with n={len(intervals_pruned)}, k={k_pruned}.')
    else:
        indices_pruned, top = list(range(len(intervals))), []
        intervals_pruned = intervals
        k_pruned = k
        T_pruned = T

        if verbose:
            print(f'Not pruning the LP as a first step.')

    if k_pruned <= 0:
        return k, [1]*k + [0]*(len(intervals)-k)

    # (2) solve the pruned instance
    # Create a new model
    model = gp.Model("interval_optimization")
    model.setParam('OutputFlag', 0)
    
    # Decision variables
    n = len(intervals_pruned)
    p = model.addVars(n, lb=0, ub=1, name="p")
    v = model.addVar(name="v")
    
    # Constraints: p sums to k
    model.addConstr(gp.quicksum(p[i] for i in range(n)) == k_pruned, "sum_p")
    
    # Add constraints for specific solving method
    add_adversary_constraints(model, p, v, intervals_pruned, T_pruned)
    
    if p_lower_bound is not None:
        # Constraints: p_i >= p_lower_bound
        for i in range(len(indices_pruned)):
            model.addConstr(p[i] >= p_lower_bound[indices_pruned[i]], f"p_lb_{i}")

    if verbose:
        print(f'Solving with {model.NumConstrs} constraints.')

    # Objective: maximize v
    model.setObjective(v, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        v_star = v.X
        p_star = np.array([p[i].X for i in range(n)])
    else:
        raise Exception('No solution found to optimization problem.')

    # (2.5) If lex_order_p != 0, re-solve with lexicographic objective.
    if lex_order_p != 0:
        model.addConstr(v == v_star, "fix_v")
        # Lexicographic weights (exponentially decreasing)
        epsilon = np.array([lex_order_p * 2 ** -(i + 1) for i in range(n)])
        model.setObjective(gp.quicksum(epsilon[i] * p[i] for i in range(n)), GRB.MAXIMIZE)
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            p_star = np.array([p[i].X for i in range(n)])
        else:
            raise Exception('No solution found to lexicographic optimization problem.')
            
        if verbose:
            print(f'Solved for lexicographic ordering on p.')

    if verbose:
        print(f'Solved with optimal value: {v_star} out of {k_pruned}.')

    # (3) post-process solution to ensure that if i dominates j, then either p_i = 1 or p_j = 0
    if postprocess:
        p_star = postprocess_solution(p_star, intervals_pruned)
        if verbose:
            print(f'Finished post-processing solution.')

    # (4) add top and bottom intervals back to the solution with p=1 and p=0 respectively
    p_out = np.zeros(len(intervals))
    for i, p_i in enumerate(p_star):
        p_out[indices_pruned[i]] = p_i
    p_out[top] = 1.

    v_out = evaluate_p(intervals, p_out, T)
    
    if verbose:
        print(f'Final value: {v_out} out of {k}.')

    return v_out, np.clip(p_out, 0, 1) 

# Given a list of intervals, a permutation of indices p_order specifying order of p_i
# solve the instance and return the optimal value and the optimal p vector.
def solve_instance_ordered(intervals, T, k, p_order, lex_order_p=0, prune=False, verbose=True, p_lower_bound=None, postprocess=False):
    # validate input
    assert all(intervals[i][0] >= intervals[i+1][0] for i in range(len(intervals)-1)), "Intervals not sorted in decreasing order of LCB."
    constraints = get_constraints(intervals)
    assert all((p_order[i] < p_order[j]) for i,j in constraints), "p_order violates constraints."

    # get inverse permutation of p_order
    p_order_inv = np.zeros(len(intervals), dtype=int)
    for i, p_i in enumerate(p_order):
        p_order_inv[p_i] = i

    def add_constraints(model, p, v, intervals, T):
        n = len(intervals)
        # Constraints: p is monotonically non-increasing with respect to indexing by p_order 
        for i in range(n-1):
            model.addConstr(p[p_order_inv[i]] >= p[p_order_inv[i+1]], f"mono_constr_{i}")
        
        # Constraints: v is adversarys worst case choice
        for i in range(T):
            # get the set of intervals that overlap with i 
            S_i = [j for j in range(i+1, len(intervals)) if intervals[j][1] >= intervals[i][0]]  
            if len(S_i) < (T - i):
                continue
            J_i = sorted(S_i, key=lambda j: -p_order[j])[:T-i] # get the T-i largest indices with respect to p_order_inv
            model.addConstr(v <= gp.quicksum(p[j] for j in range(i)) + gp.quicksum(p[j] for j in J_i), f"adv_constr_{i}")
    
    return solve_instance_base(intervals, T, k, add_constraints, lex_order_p, verbose, prune, p_lower_bound, postprocess)

# Given a list of intervals, return the optimal value and the optimal p vector.
def solve_instance_unordered(intervals, T, k, lex_order_p=0, verbose=True, prune=True, p_lower_bound=None, postprocess=True):
    # re-order intervals by decreasing number of intervals strictly below each interval breaking ties incresaing order of intervals above 
    intervals, order = sort_intervals(intervals)
    # re-order p_lower_bound to match the new order of intervals
    if p_lower_bound is not None:
        p_lower_bound = [p_lower_bound[i] for i in order]

    def add_constraints(model, p, v, intervals, T):
        # Get monotonic and non-monotonic intervals
        _, inds = partition_intervals(intervals, return_inds=True)
        if verbose:
            print('Partitioned intervals into {} subsets.'.format(len(inds)))

        # Constraints: p is monotonically non-increasing for intervals within each partition
        for p_inds in inds:
            for i in range(len(p_inds)-1):
                model.addConstr(p[p_inds[i]] >= p[p_inds[i+1]], f"mono_constr_{p_inds[i]}_{p_inds[i+1]}")

        # Constraints: v is adversarys worst case choice
        for i in range(T):
            # get the set of intervals that overlap with i 
            S_i = [j for j in range(i+1, len(intervals)) if intervals[j][1] >= intervals[i][0]]  
            if len(S_i) < T - i:
                continue
            # Split S_i into partitions and sort in reverse order
            J_i_sets = [[j for j in part if j in S_i][::-1] for part in inds]
            # Get all ways to choose (T-i) elements from J_i_sets 
            for idx, selection in enumerate(generate_prefix_selections(J_i_sets, T-i)):
                model.addConstr(v <= gp.quicksum(p[j] for j in range(i)) + gp.quicksum(p[j] for j in selection), f"adv_constr_{i}_{idx}")
    
    v, p = solve_instance_base(intervals, T, k, add_constraints, lex_order_p, verbose, prune, p_lower_bound, postprocess)
    # reorder p to original order
    p_out = np.zeros(len(intervals))
    for i, p_i in enumerate(p):
        p_out[order[i]] = p_i
    return v, p_out

# Given a list of intervals, return the optimal value and the optimal p vector.
def solve_instance_bruteforce(intervals, T, k, p_order=None, lex_order_p=0, verbose=True, prune=True, p_lower_bound=None, postprocess=True):
    if p_order is not None:
        constraints = get_constraints(intervals)
        assert all((p_order[i] < p_order[j]) for i,j in constraints), "p_order violates constraints."

    def add_constraints(model, p, v, intervals, T):
        n = len(intervals)

        if p_order is not None:
            # Constraints: p is monotonically non-increasing with respect to indexing by p_order 
            for i in range(n-1):
                model.addConstr(p[p_order[i]] >= p[p_order[i+1]], f"mono_constr_{i}")
        
        # Generate all possible permutations of theta that satisfy sigma(i) < sigma(j) for all (i,j) in constraints
        perm_constraints = get_constraints(intervals)
        perms = generate_constrained_permutations(n, perm_constraints)
        thetas = [1] * T + [0] * (n - T)
        theta_perms = [[thetas[seq[i]] for i in range(n)] for seq in perms]
        theta_perms = list(set(map(tuple, theta_perms)))

        # Constraints: v is adversarys worst case choice
        for idx, theta_p in enumerate(theta_perms):
            model.addConstr(v <= gp.quicksum(p[i] * theta_p[i] for i in range(n)), f"theta_constr_{idx}")
    
    return solve_instance_base(intervals, T, k, add_constraints, lex_order_p, verbose, prune, p_lower_bound, postprocess)

# Solve sequence of optimization problems to ensure p is monotonic in k
def solve_sequence(intervals, k, lex_order_p=0, verbose=True, prune=True, postprocess=True, return_seq=False, check_monotonicity=True):
    p_lower_bound=None
    p_seq = []
    v_seq = []
    for i in range(1, k+1):
        # Solve the instance with the current lower bound
        v, p = solve_instance_unordered(intervals, i, i, lex_order_p=lex_order_p, verbose=verbose, prune=prune, p_lower_bound=p_lower_bound, postprocess=postprocess)
        if check_monotonicity and p_lower_bound is not None:
            for j in range(len(p)):
                # Ensure that p is monotonic in k
                assert(p[j] >= p_lower_bound[j] or np.isclose(p[j], p_lower_bound[j], atol=0.002)), f"p is not monotonic for k={i} p({j})={round(p[j],3)} and (k-1)={i-1}, p({j})={round(p_lower_bound[j],3)})."
        # Update the lower bound for the next iteration
        p_lower_bound = p
        p_seq.append(p)
        v_seq.append(v)
    if return_seq:
        return v_seq, p_seq
    return v, p

# Given a list of intervals, return a list of constraints where each constraint is a tuple (i, j).
def get_constraints(intervals):
    n = len(intervals)
    # tuples where LCB_i > UCB_j => i < j
    constraints = [(i,j) for i in range(n) for j in range(n) if intervals[i][0] > intervals[j][1]]
    return constraints

# Generate all permutations of [0...(n-1)] satisfying sigma(i) < sigma(j) for all (i,j) in constraints
def generate_constrained_permutations(n, constraints):
    # Build predecessor map: pred[j] = set of nodes that must come before j
    pred = defaultdict(set)
    for i, j in constraints:
        pred[j].add(i)

    result = []
    
    def backtrack(path):
        if len(path) == n:
            result.append(path.copy())
            return
        
        # Find available nodes: not in path and all predecessors are in path
        available = []
        for node in range(n):
            if node not in path and pred[node].issubset(path):
                available.append(node)
        
        for node in available:
            backtrack(path + [node])
    
    backtrack([])
    return result

# Return permutations of indices tau, such that tau(i) = rank of i by x
def get_order_by_x(x):
    tau_inv = np.argsort(x)[::-1]
    tau = np.zeros(len(tau_inv), dtype=int)
    for i, t in enumerate(tau_inv):
        tau[t] = i
    return tau

def generate_prefix_selections(sets, m):
    """
    Given a list of lists sets and an integer m, generate every possible way
    to choose m items from the union of sets with the constraint that from
    each list we only take a prefix (i.e. the first a_i items from list i).
    
    Note: It is assumed that the total number of items available (sum of lengths) is at least m.
    """
    # For each list, the maximum allowed count is its length.
    bounds = [len(lst) for lst in sets]
    k = len(sets)
    
    # Helper function: recursively generate all vectors of length `k` that sum to `m`,
    # with the extra constraint that the i-th element is at most bounds[i].
    def rec(i, remaining, current):
        # if we are at the last list, then the last count must equal 'remaining'
        if i == k - 1:
            if 0 <= remaining <= bounds[i]:
                yield current + [remaining]
            return
        # For list i, try all possible selections from 0 to min(bounds[i], remaining)
        for x in range(0, min(bounds[i], remaining) + 1):
            yield from rec(i + 1, remaining - x, current + [x])
    
    # Generate all possible a_lists (the composition of counts)
    all_a_lists = list(rec(0, m, []))
    
    # For each composition, build the corresponding selection (by taking the prefix of each list)
    results = []
    for a_list in all_a_lists:
        selection = []
        for sublist, count in zip(sets, a_list):
            # take the prefix of length 'count'
            selection.extend(sublist[:count])
        results.append((selection))
    
    return results