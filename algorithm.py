import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import bisect

#######################################################################
#                           Optimization                              #
#######################################################################

# Given a list of intervals, return a list of constraints where each constraint is a tuple (i, j).
def get_constraints(intervals):
    n = len(intervals)
    # tuples where LCB_i > UCB_j => i < j
    constraints = [(i,j) for i in range(n) for j in range(n) if intervals[i][0] > intervals[j][1]]
    return constraints

# Given a list of intervals and a number T, prune out intervals that are always in the top T or never in the top T.
def prune_instance(intervals, T):
    n = len(intervals)
    constraints = get_constraints(intervals)
    A = [0]*len(intervals)
    B = [0]*len(intervals)
    for c in constraints:
        A[c[0]] += 1 # num intervals that i is strictly above 
        B[c[1]] += 1 # num intervals that i is strictly below
    # if A[i] >= n-T, then i must be in the top T
    top = [i for i in range(n) if A[i] >= n-T]
    # if B[i] >= T, then i is never in the top T
    bottom = [i for i in range(n) if B[i] >= T]
    # remove top and bottom from intervals
    intervals = [i for i in range(n) if i not in top+bottom]
    return intervals, top, bottom

def solve_instance_base(intervals, T, k, add_adversary_constraints, lex_order_p=0, verbose=True, prune=True, p_lower_bound=None):
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
    # Decision variables
    n = len(intervals_pruned)
    p = cp.Variable(n, nonneg=True)
    v = cp.Variable()

    # Constraints: p sums to k
    cp_constraints = [cp.sum(p) == k_pruned]
    # Constraints: all p_i are in [0, 1]
    cp_constraints += [p[i] <= 1 for i in range(n)]
    # Add constraints for specific solving method
    cp_constraints += add_adversary_constraints(p, v, intervals_pruned, T_pruned)
    if p_lower_bound is not None:
        # Constraints: p_i >= p_lower_bound
        cp_constraints += [p[i] >= p_lower_bound[indices_pruned[i]] for i in range(len(indices_pruned))]

    if verbose:
        print(f'Solving with {len(cp_constraints)} constraints.')

    # Objective: maximize v
    objective = cp.Maximize(v)
    problem = cp.Problem(objective, cp_constraints)
    problem.solve()

    v_star = v.value
    p_star = p.value

    # (2.5) If lex_order_p != 0, re-solve with lexicographic objective.
    if lex_order_p != 0:
        cp_constraints.append(v == v_star)  # Fix v to v_opt
        # Lexicographic weights (exponentially decreasing)
        epsilon = np.array([lex_order_p * 2 ** -(i + 1) for i in range(n)])
        problem = cp.Problem(cp.Maximize(epsilon @ p), cp_constraints)
        problem.solve()
        if verbose:
            print(f'Solved for lexicographic ordering on p.')

    if verbose:
        print(f'Solved with optimal value: {v_star} out of {k_pruned}.')

    # (3) post-process solution to ensure that if i dominates j, then either p_i = 1 or p_j = 0
    p_star = postprocess_solution(p_star, intervals_pruned)
    if verbose:
        print(f'Finished post-processing solution.')

    # (4) add top and bottom intervals back to the solution with p=1 and p=0 respectively
    v_out = v_star + len(top)
    p_out = np.zeros(len(intervals))
    for i, p_i in enumerate(p_star):
        p_out[indices_pruned[i]] = p_i
    p_out[top] = 1
    return v_out, p_out

# Given a list of intervals, a permutation of indices p_order specifying order of p_i
# solve the instance and return the optimal value and the optimal p vector.
def solve_instance_ordered(intervals, T, k, p_order, lex_order_p=0, prune=False, verbose=True, p_lower_bound=None):
    # validate input
    assert all(intervals[i][0] >= intervals[i+1][0] for i in range(len(intervals)-1)), "Intervals not sorted in decreasing order of LCB."
    constraints = get_constraints(intervals)
    assert all((p_order[i] < p_order[j]) for i,j in constraints), "p_order violates constraints."

    # get inverse permutation of p_order
    p_order_inv = np.zeros(len(intervals), dtype=int)
    for i, p_i in enumerate(p_order):
        p_order_inv[p_i] = i

    def add_constraints(p, v, intervals, T):
        n = len(intervals)
        # Constraints: p is monotonically non-increasing with respect to indexing by p_order 
        cp_constraints = [p[p_order_inv[i]] >= p[p_order_inv[i+1]] for i in range(n-1)]
        # Constraints: v is adversarys worst case choice
        for i in range(T):
            # get the set of intervals that overlap with i 
            S_i = [j for j in range(i+1, len(intervals)) if intervals[j][1] >= intervals[i][0]]  
            if len(S_i) < (T - i):
                continue
            J_i = sorted(S_i, key=lambda j: -p_order[j])[:T-i] # get the T-i largest indices with respect to p_order_inv
            cp_constraints += [v <= cp.sum([p[j] for j in range(i)]) + cp.sum([p[j] for j in J_i])]
        return cp_constraints
    
    return solve_instance_base(intervals, T, k, add_constraints, lex_order_p, verbose, prune, p_lower_bound)

# Given a list of intervals, return the optimal value and the optimal p vector.
def solve_instance_unordered(intervals, T, k, p_order, lex_order_p=0, verbose=True, prune=True, p_lower_bound=None):
    # re-order intervals by number above and below each interval
    intervals, order = sort_intervals(intervals)

    def add_constraints(p, v, intervals, T):
        # Get monotonic and non-monotonic intervals
        _, inds = partition_intervals(intervals, return_inds=True)

        cp_constraints = []
        # Constraints: p is monotonically non-increasing for intervals within each partition
        for p_inds in inds:
            cp_constraints += [p[p_inds[i]] >= p[p_inds[i+1]] for i in range(len(p_inds)-1)]

        # Constraints: v is adversarys worst case choice
        for i in range(T):
            # get the set of intervals that overlap with i 
            S_i = [j for j in range(i+1, len(intervals)) if intervals[j][1] >= intervals[i][0]]  
            if len(S_i) < T - i:
                continue
            # Split S_i into partitions and sort in reverse order
            J_i_sets = [[j for j in part if j in S_i][::-1] for part in inds]
            # Get all ways to choose (T-i) elements from J_i_sets 
            for selection in generate_prefix_selections(J_i_sets, T-i):
                cp_constraints += [v <= cp.sum([p[j] for j in range(i)]) + cp.sum([p[j] for j in selection])]
        return cp_constraints
    
    v, p = solve_instance_base(intervals, T, k, add_constraints, lex_order_p, verbose, prune, p_lower_bound)
    # reorder p to original order
    p_out = np.zeros(len(intervals))
    for i, p_i in enumerate(p):
        p_out[order[i]] = p_i
    return v, p_out

# Given a list of intervals, return the optimal value and the optimal p vector.
def solve_instance_bruteforce(intervals, T, k, p_order=None, lex_order_p=0, verbose=True, prune=True, p_lower_bound=None):
    if p_order is not None:
        constraints = get_constraints(intervals)
        assert all((p_order[i] < p_order[j]) for i,j in constraints), "p_order violates constraints."

    def add_constraints(p, v, intervals, T):
        n = len(intervals)

        if p_order is not None:
            # Constraints: p is monotonically non-increasing with respect to indexing by p_order 
            cp_constraints = [p[p_order[i]] >= p[p_order[i+1]] for i in range(n-1)]
        else:
            cp_constraints = []
        
        # Generate all possible permutations of theta that satisfy sigma(i) < sigma(j) for all (i,j) in constraints
        perm_constraints = get_constraints(intervals)
        perms = generate_constrained_permutations(n, perm_constraints)
        thetas = [1] * T + [0] * (n - T)
        theta_perms = [[thetas[seq[i]] for i in range(n)] for seq in perms]
        theta_perms = list(set(map(tuple, theta_perms)))

        # Constraints: v is adversarys worst case choice
        for theta_p in theta_perms:
            cp_constraints += [v <= cp.sum([p[i] * theta_p[i] for i in range(n)])]
        
        return cp_constraints

    return solve_instance_base(intervals, T, k, add_constraints, lex_order_p, verbose, prune, p_lower_bound)

# Ensure that if i dominates j, then either p_i = 1 or p_j = 0
def postprocess_solution(p, intervals):
    items = list(zip(p, intervals, range(len(p))))
    
    # Sort items by the upper bound of intervals (u_i)
    items.sort(key=lambda x: x[1][1])
    
    # Extract the sorted probabilities, intervals, and original indices
    sorted_p, sorted_intervals, original_indices = zip(*items)
    
    # Convert sorted_p to a list for mutability
    sorted_p = list(sorted_p)
    
    n = len(sorted_p)
    
    # Iterate over each element in the sorted list
    for b in range(n):
        if sorted_p[b] == 0:
            break
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
#                              Helpers                                #
#######################################################################

def plot_intervals(intervals, order=None, x=None):
    if order is not None:
        intervals = [intervals[i] for i in order]
    n = len(intervals)
    _, ax = plt.subplots()
    for i, (a,b) in enumerate(intervals):
        ax.plot([i,i], [a,b], 'k-')
    
    if x is not None:
        # plot x values as dots
        for i, x_i in enumerate(x):
            ax.plot(i, x_i, 'ro', markersize=5)

    # smart x ticks so that not too crowded
    if n > 10:
        step = max(1, n // 10)
        # round step to closest multiple of 5
        step = 5 * max(round(step/5), 1)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([f'{i}' for i in range(0, n, step)])
    else:
        ax.set_xticks(range(n))
        ax.set_xticklabels([f'{i}' for i in range(n)])

    ax.set_xlabel('Interval')
    ax.set_ylabel('Value')

    return ax

# Swiss-NSF algorithm, given a list of intervals and a number k, return p vector
def swiss_nsf(intervals, x, k):
    # funding line is point estimate of the kth item
    line = sorted(x, reverse=True)[k-1]

    # intervals strictly above funding line
    above = [i for i in range(len(intervals)) if intervals[i][0] > line]
    # intervals strictly below funding line
    below = [i for i in range(len(intervals)) if intervals[i][1] < line]
    # intervals that overlap with funding line
    overlap = [i for i in range(len(intervals)) if intervals[i][0] <= line <= intervals[i][1]]

    k_rand = k - len(above)
    if k_rand == 0:
        return [1]*k + [0]*(len(intervals)-k)

    p = [0]*len(intervals)
    for i in above:
        p[i] = 1
    for i in below:
        p[i] = 0
    for i in overlap:
        p[i] = 1. * k_rand / len(overlap)
    return p

# Given a choice of p vector, evaluate the worst case expected value of sampling according to p over rankings
# consistent with ordering of intervals.
def evaluate_p(intervals, p, T):
    # sort intervals in decreasing order of LCB
    ordering = np.argsort([i[0] for i in intervals])[::-1]
    intervals = [intervals[i] for i in ordering]
    p = [p[i] for i in ordering]
    n = len(intervals)

    v = T
    for i in range(T):
        # get the values of p for set of intervals that overlap with i 
        S_i = [p[j] for j in range(i+1, n) if intervals[j][1] >= intervals[i][0]]  
        if len(S_i) < T - i:
            continue
        v_i = sum(p[:i]) + sum(sorted(S_i)[:T-i])
        if v_i < v:
            v = v_i
    return v

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

# Sort intervals by decreasing # of intervals they are strictly above, breaking ties by # of intervals they are strictly below
def sort_intervals(intervals, return_AB=False):
    n = len(intervals)
    constraints = get_constraints(intervals)
    A = [0]*len(intervals)
    B = [0]*len(intervals)
    for c in constraints:
        A[c[0]] += 1 # num intervals that i is strictly above 
        B[c[1]] += 1 # num intervals that i is strictly below
    # sort 1 to n by decreasing A breaking ties with increasing B
    order = sorted(range(n), key=lambda i: (-A[i], B[i]))
    if return_AB:
        return [intervals[i] for i in order], order, A, B
    return [intervals[i] for i in order], order

# Return permutations of indices tau, such that tau(i) = rank of i by x
def get_order_by_x(x):
    tau_inv = np.argsort(x)[::-1]
    tau = np.zeros(len(tau_inv), dtype=int)
    for i, t in enumerate(tau_inv):
        tau[t] = i
    return tau

# Given a list of intervals, partition them into disjoint subsets such that all intervals in a subset are monotonically ordered.
def partition_intervals(intervals, return_inds=False):
    # Sort intervals by decreasing # of intervals they are strictly above, breaking ties by # of intervals they are strictly below
    _, order, _, B = sort_intervals(intervals, return_AB=True)
    B = [-b for b in B]
    # Use a list to track the last B of each subset
    subsets = []
    # To track the actual subsets, we maintain a list of lists
    actual_subsets = []
    actual_subsets_ind = []
    for j in order:
        B_j = B[j]
        # Find the first index in subsets where last_u >= u
        idx = bisect.bisect_left(subsets, B_j)
        if idx < len(subsets):
            # Replace the subset's last u with current u
            subsets[idx] = B_j
            actual_subsets[idx].append(intervals[j])
            actual_subsets_ind[idx].append(j)
        else:
            # Create a new subset
            subsets.append(B_j)
            actual_subsets.append([intervals[j]])
            actual_subsets_ind.append([j])  
    if return_inds:
        # Return the indices of the intervals in the original order
        return actual_subsets, actual_subsets_ind

    # Return the actual subsets
    return actual_subsets

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
