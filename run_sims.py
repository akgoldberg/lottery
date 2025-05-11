from generate_intervals import *
from cutting_plane import solve_problem, solve_with_monotonicity
from helpers import swiss_nsf, top_k
import time 
import json
import ast

def parse_results_file(filepath):
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ': ' not in line:
                continue  # skip malformed lines
            key, value = line.split(': ', 1)
            try:
                # Safely evaluate Python literals (like lists, dicts, numbers)
                results[key.strip()] = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError):
                # Fallback: store as raw string
                results[key.strip()] = value.strip()
    return results

def run_and_save_results(x, intervals, k, filename, run_monotonicity=True):
    """
    Run the cutting plane algorithm and save results to a file.
    """
    p_opt, v_opt, info = solve_problem(intervals, k, verbose=True)
    if run_monotonicity:
        p_seq, v_seq, info_seq = solve_with_monotonicity(intervals, k, verbose=False)
    p_swiss_seq = [swiss_nsf(intervals, x, i) for i in range(1,k+1)]
    p_top_k = top_k(x, k)

    # Save results
    with open(filename, 'w') as f:
        f.write('p_opt: {}\n'.format(p_opt))
        f.write('v_opt: {}\n'.format(v_opt))
        f.write('info: {}\n'.format(info))
        if run_monotonicity:
            f.write('p_seq: {}\n'.format(p_seq))
            f.write('v_seq: {}\n'.format(v_seq))
            f.write('info_seq: {}\n'.format(info_seq))
        f.write('p_swiss: {}\n'.format(p_swiss_seq))
        f.write('top_k: {}\n'.format(p_top_k))

def run_case_studies():
    print('===========Running Swiss NSF===========')
    t = time.time()
    # Load Swiss NSF data
    x, intervals, intervals90, half_intervals, half_intervals90 = load_swiss_nsf()
    k = 106
    run_and_save_results(x, intervals, k, 'res/swiss_nsf_results.txt')
    print('Completed Swiss NSF in: ', time.time() - t, ' seconds')

    print('===========Running NeurIPS LOO===========')
    t = time.time()
    # Load NeurIPS data
    x, intervals, decisions = load_neurips_leaveoneout()
    k = min(decisions.value_counts()) # number of reject decisions
    run_and_save_results(x, intervals, k, 'res/neuripsloo_results.txt')
    print('Completed Neurips LOO in: ', time.time() - t, ' seconds')

    print('===========Running NeurIPS Gaussian===========')
    t = time.time()
    # Load NeurIPS data
    x, intervals50, intervals95, decisions = load_neurips_gaussian_model()
    k = min(decisions.value_counts()) # number of reject decisions
    run_and_save_results(x, intervals50, k, 'res/neuripsgaussian_results.txt')
    print('Completed Neurips Gaussian in: ', time.time() - t, ' seconds')

    print('===========Running ICLR LOO===========')
    t = time.time()
    # Load ICLR data
    x, intervals, decisions = load_neurips_leaveoneout()
    k = min(decisions.value_counts()) # number of reject decisions
    run_and_save_results(x, intervals, k, 'res/iclrloo_results.txt', run_monotonicity=False)
    print('Completed ICLR LOO in: ', time.time() - t, ' seconds')

    print('===========Running ICLR Gaussian===========')
    t = time.time()
    # Load ICLR data
    x, intervals50, intervals95, decisions = load_neurips_gaussian_model()
    k = min(decisions.value_counts()) # number of reject decisions
    run_and_save_results(x, intervals50, k, 'res/iclrgaussian_results.txt', run_monotonicity=False)
    print('Completed ICLR Gaussian in: ', time.time() - t, ' seconds')

def run_random_ablations(n, n_iters=20):
    ks = [n // 100, n // 20, n // 10, n // 5, n // 2]

    random_intervals = [generate_uniform_intervals(n) for _ in range(n_iters)]
    results = []

    for k in ks:
        print(f'Running ablation study for k={k}...')
        for I in random_intervals:
            result_entry = {"k": k, "intervals": I}

            # Solve with all optimizations
            _, _, info_all = solve_problem(I, k, use_symmetry=True, add_monotonicity_constraints=True, init_prune=True)
            result_entry["info_all"] = info_all

            # Solve with only symmetry + monotonicity
            _, _, info_noprune = solve_problem(I, k, use_symmetry=True, add_monotonicity_constraints=True, init_prune=False)
            result_entry["info_noprune"] = info_noprune

            # Solve with only pruning
            _, _, info_pruneonly = solve_problem(I, k, use_symmetry=False, add_monotonicity_constraints=False, init_prune=True)
            result_entry["info_pruneonly"] = info_pruneonly

            # Solve with no optimizations
            _, _, info_none = solve_problem(I, k, use_symmetry=False, add_monotonicity_constraints=False, init_prune=False)
            result_entry["info_none"] = info_none

            results.append(result_entry)
        
    # Save results to file
    with open('res/random_ablations_results.json', 'w') as f:
        json.dump(results, f, indent=4)


def run_data_ablations(data):
    results = []
    # Load data
    if data == 'neurips':
        _, I, _, decisions = load_neurips_gaussian_model()
        k = min(decisions.value_counts()) # number of reject decisions
        ks = [k // 100, k // 50, k // 20, k // 10, k // 5, k // 2, k // 1, int(k*1.2), int(k*1.5), int(k*2), int(k*3)]
        max_k = 400
    elif data == 'iclr':
        _, I, _, decisions = load_iclr_gaussian_model()
        k = min(decisions.value_counts()) # number of reject decisions
        ks = [k // 100, k // 50, k // 20, k // 10, k // 5, k // 2, k // 1]
        max_k = 200
    elif data == 'swissnsf':
        _, I, _, _, _ = load_swiss_nsf()
        k = 106
        ks = [k // 20, k // 10, k // 5, k // 2, k // 1, int(1.2 * k), int(1.5*k)]
        max_k = len(I)
    else:
        raise ValueError("Invalid data type. Choose either 'neurips' or 'iclr'.")

    for k in ks:
        print(f'Running ablation study for k={k}...')
        result_entry = {"k": k, "data": data}
        # Solve with all optimizations
        _, _, info_all = solve_problem(I, k, use_symmetry=True, add_monotonicity_constraints=True, init_prune=True)
        result_entry["info_all"] = info_all
        # Solve with only symmetry + monotonicity
        _, _, info_noprune = solve_problem(I, k, use_symmetry=True, add_monotonicity_constraints=True, init_prune=False)
        result_entry["info_noprune"] = info_noprune

        if k <= max_k:
            # Solve with only pruning
            _, _, info_pruneonly = solve_problem(I, k, use_symmetry=False, add_monotonicity_constraints=False, init_prune=True)
            result_entry["info_pruneonly"] = info_pruneonly
        if k <= max_k:
            # Solve with no optimizations
            _, _, info_none = solve_problem(I, k, use_symmetry=False, add_monotonicity_constraints=False, init_prune=False)
            result_entry["info_none"] = info_none

        results.append(result_entry)

    # Save results to file
    with open(f'res/{data}_ablations_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # print('===========Running All Case Studies===========')
    # run_case_studies()

    print('===========Running Ablations===========')
    run_data_ablations('neurips')

