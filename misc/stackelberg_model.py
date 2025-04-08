import cvxpy as cp
import numpy as np
import itertools

# constraints: list of n tuples [a_i,b_i] representing a_i <= sigma_i <= b_i
def solve_monotonic_01(n, k, constraints, lex_order_p=-1, verbose=False):
    """Solve the optimization problem 0-1 utility, monotonic intervals."""
    # Decision variables
    p = cp.Variable(n, nonneg=True)
    v = cp.Variable()

    # Constraints: p sums to k
    cp_constraints = [cp.sum(p) == k]

    # Constraints: all p_i are in [0, 1]
    cp_constraints += [p[i] <= 1 for i in range(n)]

    # Constraints: p is monotone
    cp_constraints += [p[i] >= p[i+1] for i in range(n-1)]

    adversary_constraints = []
    adv_constraints_info = []  # list to store (j, m_j, rhs_expr)
    
    # Build adversary's worst-case selection constraints.
    for j in range(k):
        m_j = constraints[j][1]
        if m_j >= k-1:
            rhs_expr = cp.sum(p[0:j]) + cp.sum(p[m_j+1-(k-j):m_j+1])
            adversary_constraints.append(v <= rhs_expr)
            # Save the info so we can later check tightness.
            adv_constraints_info.append((j, m_j, rhs_expr))
    
    cp_constraints += adversary_constraints

    # Objective: maximize v
    problem = cp.Problem(cp.Maximize(v), cp_constraints)
    problem.solve()
    v_opt = v.value

    # If lex_order_p != 0, re-solve with lexicographic objective.
    if lex_order_p != 0:
        cp_constraints.append(v == v_opt)  # Fix v to v_opt
        # Lexicographic weights (exponentially decreasing)
        epsilon = np.array([lex_order_p * 2 ** -(i + 1) for i in range(n)])
        problem = cp.Problem(cp.Maximize(epsilon @ p), cp_constraints)
        problem.solve()

    # After solving,check which adversary constraints are tight.
    tol = 1e-6
    tight_constraints = []
    for (j, m_j, rhs_expr) in adv_constraints_info:
        # Evaluate the right-hand side expression.
        rhs_val = rhs_expr.value
        lhs_val = v.value
        if abs(lhs_val - rhs_val) <= tol:
            tight_constraints.append((j))
            
    return p.value, v_opt, tight_constraints


"""Get all permutations of length n with one item shifted to at a different position of at most max_shift"""
def generate_single_insertion_permutations(n, max_shift, return_distances=False):
    seq = list(range(n))
    insertions = [seq]
    distances = [0]
    # move i to position r within max_shift positions
    for i in range(n):
        for r in range(max(0, i - max_shift), min(n, i + max_shift + 1)):
            if i != r:
                new_seq = seq.copy()
                new_seq.pop(i)
                new_seq.insert(r, i)
                insertions.append(new_seq)                
    
    insertions = list(set(map(tuple, insertions)))
    if return_distances:
        distances = [max(abs(i - r) for i, r in enumerate(seq)) for seq in insertions]
        return insertions, distances
    return insertions

"""Get all permutations of length n with m items shifted to a different position of at most D"""
def generate_mD_insertion_permutations(n, m, D):
    def insert_items(orig, indices, positions):
        temp = orig[:]
        items = [orig[i] for i in indices]
        for i in sorted(indices, reverse=True):
            del temp[i]
        for item, pos in zip(items, positions):
            temp.insert(pos, item)
        return temp
    
    orig = list(range(n))
    out = set()
    for indices in itertools.combinations(range(n), m):
        for shifts in itertools.permutations(range(-D, D+1), m):
            positions = [min(max(i + s, 0), n-1) for i, s in zip(indices, shifts)]
            perm = insert_items(orig, indices, positions)
            out.add(tuple(perm))
    return list(out)


def get_all_linf_distances(n):
    """Get all L_inf distances between identity and all permutations of {1, ..., n}."""
    identity = list(range(n))
    distances = [(max(np.abs(np.array(perm) - np.array(identity))),perm) for perm in itertools.permutations(identity)]
    return distances

def generate_linf_permutations(n, D, distance_dict):
    return [perm for dist, perm in distance_dict if dist <= D]   

def kendall_tau_distance(perm, identity):
    """Compute Kendall Tau distance between perm and identity."""
    n = len(perm)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if (perm[i] < perm[j] and identity[i] > identity[j]) or (perm[i] > perm[j] and identity[i] < identity[j]):
                count += 1
    return count

def get_all_kendall_tau_distances(n):
    """Get all Kendall Tau distances between identity and all permutations of {1, ..., n}."""
    identity = list(range(n))
    distances = [(kendall_tau_distance(perm, identity), perm) for perm in itertools.permutations(identity)]
    return distances

"""Generate all permutations of {0,2,...,n-1} at Kendall tau distance <=D from identity."""
def generate_kendall_tau_permutations(n, D, distance_dict=None):
    if distance_dict is not None:
        perms = [perm for dist, perm in distance_dict if dist <= D]
        return perms
    
    # Start with the base permutation [0] and 0 inversions
    permutations = [([0], 0)]
    
    for i in range(1, n):
        new_perms = []
        for perm, current_count in permutations:
            # Insert new element i in all possible positions
            for pos in range(len(perm) + 1):
                # Calculate new inversion count using the formula:
                # added = (element value) - insertion position
                added_inversions = i - pos
                new_count = current_count + added_inversions
                
                if new_count <= D:
                    new_perm = perm[:pos] + [i] + perm[pos:]
                    new_perms.append((new_perm, new_count))
        
        permutations = new_perms
    
    # Return only the permutations (without inversion counts)
    perms = [perm for perm, count in permutations]
    # dedupe the permutations
    perms = list(set(map(tuple, perms)))
    return perms

"""Efficiently count number of permutations of {1,2,...,n} at Kendall tau distance <=D from identity."""
def count_kt_permutations(n,D): 
    def count_kt_permutations_exact(n: int, D: int) -> int:
        max_inversions = n * (n - 1) // 2
        if D < 0 or D > max_inversions:
            return 0
        
        # dp[i] represents number of permutations with i inversions
        dp = [0] * (D + 1)
        dp[0] = 1  # Base case: identity permutation
        
        for k in range(2, n+1):
            new_dp = [0] * (k*(k-1)//2 + 1)
            prefix = [0] * (len(dp) + 1)
            
            # Compute prefix sums for previous dp array
            for i in range(len(dp)):
                prefix[i+1] = prefix[i] + dp[i]
            
            # Calculate new counts using sliding window
            for j in range(len(new_dp)):
                max_add = k - 1
                lower = max(0, j - max_add)
                upper = min(j, len(dp)-1)
                
                if lower > upper:
                    new_dp[j] = 0
                else:
                    new_dp[j] = prefix[upper+1] - prefix[lower]
            
            # Update dp array and limit size to current maximum needed
            current_max = k*(k-1)//2
            dp = new_dp[:min(current_max, D)+1]
        
        return dp[D] if D < len(dp) else 0

    return sum(count_kt_permutations_exact(n, d) for d in range(D+1))

def spearmans_footrule_distance(perm, identity):
    """Compute Spearman's Footrule distance between perm and identity."""
    return sum(abs(i - perm[i]) for i in identity)

def get_all_spearmans_footrule_distances(n):
    """Get all Spearman's Footrule distances between identity and all permutations of {1, ..., n}."""
    identity = list(range(n))
    distances = [(spearmans_footrule_distance(perm, identity), perm) for perm in itertools.permutations(identity)]
    return distances

def generate_spearmans_footrule_permutations(n, D, distance_dict=None):
    if distance_dict is not None:
        perms = [perm for dist, perm in distance_dict if dist <= D]
        return perms
    
    """Generate all permutations of {1, ..., n} within Spearman's Footrule distance D."""
    identity = list(range(n))
    perms = []

    for perm in itertools.permutations(identity):
        if spearmans_footrule_distance(perm, identity) <= D:
            perms.append(perm)
    return perms

'''
Get optimal strategy for the decision maker in the Stackelberg game.

Parameters:
- thetas: list of n scores, positional scoring rule for each rank
- k: number of items to select
- permutations: list of permutations of n items (errors)
- lex_order_p: if 1, order the probabilities lexicographically, if 0, do not order, if -1, order in reverse lexicographic order
- return_tight_constraints: if True, return tight constraints (worst-case permutations)
- monotone_p: if True, enforce monotonicity on p
- verbose: if True, print intermediate results

Returns:
- v*: optimal value
- p: list of n optimal marginal probabilities for selecting each rank 
- tight_constraints: list of tight constraints (worst-case permutations)
'''
def get_optimal_strategy(thetas, k, permutations, lex_order_p=1, return_tight_constraints=False, monotone_p=False, verbose=False):
    n = len(thetas)

    # Decision variables
    v = cp.Variable()
    p = cp.Variable(n, nonneg=True)

    # Generate adversarial permutations
    theta_perms = [[thetas[seq[i]] for i in range(n)] for seq in permutations]
    # dedupe the permutations
    theta_perms = list(set(map(tuple, theta_perms)))

    # Constraints: adversary's worst-case selection
    constraints = [v - cp.matmul(theta_p, p) <= 0 for theta_p in theta_perms]

    # Constraints: p sums to k
    constraints.append(cp.sum(p) == k)

    # Constraints: all p_i are in [0, 1]
    constraints += [p[i] <= 1 for i in range(n)]

    if monotone_p:
        # Constraints: p is monotone
        constraints += [p[i] >= p[i+1] for i in range(n-1)]

    # **Step 1: Solve for v* without modifying p**
    problem = cp.Problem(cp.Maximize(v), constraints)
    problem.solve()
    v_star = v.value  # Store optimal v

    if verbose:
        print(f"Optimal v*: {v_star:.3f}")

    if lex_order_p != 0:
            # **Step 2: Solve for lexicographically largest p with v fixed**
            constraints.append(v == v_star)  # Fix v to v*

            # Lexicographic priority weights (exponentially decreasing)
            epsilon = np.array([lex_order_p * 2 ** -(i + 1) for i in range(n)])

            # New objective: maximize lexicographic p
            problem = cp.Problem(cp.Maximize(epsilon @ p), constraints)
            problem.solve()

            if verbose:
                print("Lexicographically optimal p:", "[", ", ".join(map(str, np.round(p.value, 3))), "]")

    p_out = p.value
    v_out = np.round(v_star, 3)

    if return_tight_constraints:
        # Get tight constraints (worst-case permutations)
        tight_constraints = [theta_p for theta_p in theta_perms if np.isclose(v_star - np.dot(theta_p, p_out), 0)]
    else:
        tight_constraints = None

    return v_star, p_out, tight_constraints

def get_optimal_strategy_with_costs(thetas, k, permutations, cost, lex_order_p=True, verbose=False):
    n = len(thetas)

    theta_perms = [[thetas[seq[i]] for i in range(n)] for seq in permutations]
    
    # check if cost is a function
    if callable(cost):
        cost_perms = [cost(seq) for seq in permutations]
    # check if cost is a list
    elif isinstance(cost, list):
        cost_perms = cost
    else:
        raise ValueError("Cost must be a function or a list")

    unique_costs = set(cost_perms)
    
    v_opt = -np.inf
    p_opt = None

    # solve an LP for each cost group
    for c in unique_costs:
        # Decision variables
        v = cp.Variable()
        p = cp.Variable(n, nonneg=True)

        # Constraints: adversary's worst-case selection
        constraints = [v + c - cp.matmul(theta_p, p) - cost_p <= 0 for theta_p, cost_p in zip(theta_perms, cost_perms)]

        # Constraints: p sums to k
        constraints.append(cp.sum(p) == k)

        # Constraints: all p_i are in [0, 1]
        constraints += [p[i] <= 1 for i in range(n)]

        # **Solve for v* **
        problem = cp.Problem(cp.Maximize(v), constraints)
        problem.solve()
        v_star = v.value
        p_star = p.value

        # print out LP
        if verbose:
            print(f"Cost: {c}")
            # print(problem)
            print(f"v*: {v_star:.3f}")
            print("p*:", "[", ", ".join(map(str, np.round(p_star, 3))), "]")

        # Check if LP was infeasible
        if v_star is None:
            continue
        
        # **Solve for lexicographically largest p with v fixed**
        if lex_order_p:
            constraints.append(v == v_star)
            # Lexicographic priority weights (exponentially decreasing)
            epsilon = np.array([2 ** -(i + 1) for i in range(n)])
            # New objective: maximize lexicographic p
            problem = cp.Problem(cp.Maximize(epsilon @ p), constraints)
            problem.solve()
            p_star = p.value

        if v_star > v_opt:
            v_opt = v_star
            p_opt = p_star

    return v_opt, p_opt

def evaluate_strategy(thetas, p, generate_permutations):
    """Evaluates the worst-case sum given a probability distribution p and returns the worst-case insertion."""
    n = len(thetas)

    permutations = generate_permutations(n)
    # generate thetas for each permutation
    theta_perms = [[thetas[seq[i]] for i in range(n)] for seq in permutations] 

    # take dot product of p and theta_perms (get value and index)
    values = [np.dot(p, theta_p) for theta_p in theta_perms]
    worst_case_value = min(values)
    worst_case_seq = permutations[np.argmin(values)]
   
    return worst_case_value, worst_case_seq