import bisect
import numpy as np

# Given a list of intervals and a number T, prune out intervals that are always in the top T or never in the top T.
def prune_instance(intervals, T):
    n = len(intervals)
    if n == 0:
        return [], [], []
    
    # Extract lower and upper bounds of intervals
    lowers = [interval[0] for interval in intervals]
    uppers = [interval[1] for interval in intervals]
    
    # Sort the bounds for efficient binary search
    sorted_upper = sorted(uppers)
    sorted_lower = sorted(lowers)
    
    # Precompute A and B arrays using binary search
    A = []
    for j in range(n):
        u_j = intervals[j][1]
        # Count intervals with lower > u_j
        A.append(len(sorted_lower) - bisect.bisect_right(sorted_lower, u_j))

    B = []
    for i in range(n):
        l_i = intervals[i][0]
        # Count intervals with upper < l_i
        B.append(bisect.bisect_left(sorted_upper, l_i))
    
    # Identify indices that are always in top T or never in top T
    top = [i for i in range(n) if B[i] >= n - T]
    bottom = [i for i in range(n) if A[i] >= T]
    
    # Prune the intervals, keeping those not in top or bottom
    pruned_intervals = [i for i in range(n) if i not in top and i not in bottom]
    
    return pruned_intervals, top, bottom

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

# Given a list of intervals, partition them into disjoint subsets such that all intervals in a subset are monotonically ordered.
def partition_intervals(intervals, return_inds=False):
    # Sort intervals by decreasing # of intervals they are strictly above, breaking ties by # of intervals they are strictly below
    _, order, A, _ = sort_intervals(intervals, return_AB=True)
    # Use a list to track the last B of each subset
    subsets = []
    # To track the actual subsets, we maintain a list of lists
    actual_subsets = []
    actual_subsets_ind = []
    for j in order:
        A_j = -A[j]
        # Find the first index in subsets where last_u >= u
        idx = bisect.bisect_left(subsets, A_j)
        if idx < len(subsets):
            # Replace the subset's last u with current u
            subsets[idx] = A_j
            actual_subsets[idx].append(intervals[j])
            actual_subsets_ind[idx].append(j)
        else:
            # Create a new subset
            subsets.append(A_j)
            actual_subsets.append([intervals[j]])
            actual_subsets_ind.append([j])  
   
    if return_inds:
        # Return the indices of the intervals in the original order
        return actual_subsets, actual_subsets_ind

    # Return the actual subsets
    return actual_subsets

# Return partition of intervals such that within each partition all intervals have the same A and same B
def get_symmetric_intervals(intervals):
    # Sort intervals by A and B
    _, order, A, B = sort_intervals(intervals, return_AB=True)

    # Group intervals by their A and B values
    partitions = {}
    for idx in order:
        key = (A[idx], B[idx])
        if key not in partitions:
            partitions[key] = []
        partitions[key].append(idx)

    # Convert the dictionary values to a list of partitions
    partitioned_intervals = list(partitions.values())
    # return the indices of partions
    return partitioned_intervals

# Sort intervals by decreasing # of intervals they are strictly above, breaking ties by # of intervals they are strictly below
def sort_intervals(intervals, return_AB=False):
    n = len(intervals)
    if n == 0:
        return ([], [], [], []) if return_AB else ([], [])
    
    # Extract lower and upper bounds of intervals
    lowers = [interval[0] for interval in intervals]
    uppers = [interval[1] for interval in intervals]
    
    # Sort bounds for efficient binary search
    sorted_upper = sorted(uppers)
    sorted_lower = sorted(lowers)
    
    # Compute A[i]: number of intervals strictly above i's lower bound
    A = [len(sorted_lower) - bisect.bisect_right(sorted_lower, upper) for upper in uppers] 
    # Compute B[j]: number of intervals strictly below j's upper bound
    B = [bisect.bisect_left(sorted_upper, lower) for lower in lowers]
    
    # Sort indices by (-B[i], A[i]) criteria
    order = sorted(range(n), key=lambda i: (-B[i], A[i]))
    
    # Prepare results based on return_AB flag
    sorted_intervals = [intervals[i] for i in order]
    if return_AB:
        return sorted_intervals, order, A, B
    return sorted_intervals, order

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

#######################################################################
#                           Verify Axioms                             #
#######################################################################

# Given a list of intervals and a p vector, verify that the solution is post-hoc valid.
# This means that if i dominates j, then either p_i = 1 or p_j = 0.
def verify_posthoc_validity(intervals, p, raise_error=False):
    items = list(zip(p, intervals, range(len(p))))
    
    # Sort items by ascending order of UCB
    items.sort(key=lambda x: x[1][1])

     # Extract the sorted probabilities, intervals, and original indices
    sorted_p, sorted_intervals, original_indices = zip(*items)

    n = len(sorted_p)
    
    # Iterate over each element in the sorted list
    for b in range(n):
        if sorted_p[b] == 0:
            continue
        for a in range(n-1, b, -1):
            if sorted_intervals[a][0] > sorted_intervals[b][1] and sorted_p[a] < 1:
                if raise_error:
                    raise('Invalid solution: p[{a}] < 1 and p[{b}] > 0, but UCB[{a}] > LCB[{b}].')
                return False 
    if raise_error:
        print('Valid solution: p is post-hoc valid.')
    return True

def verify_monotonicity_in_k(pseq, raise_error=False):
    """
    Verify that the sequence of p vectors is monotonic in k.
    This means that for each i, p[i] >= p[i-1] for all i.
    """
    n = len(pseq)
    for i in range(1, n):
        p_lower_bound = pseq[i-1]
        p = pseq[i]
        for j in range(len(p)):
             if not (p[j] >= p_lower_bound[j] or np.isclose(p[j], p_lower_bound[j], atol=0.002)):
                if raise_error:
                    raise(f"p is not monotonic for k={i} p({j})={round(p[j],3)} and (k-1)={i-1}, p({j})={round(p_lower_bound[j],3)}.")
                return False

    # If we reach here, the solution is valid
    if raise_error:
        print('Valid solution: p is monotonic in k.')
    return True

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

