from generate_intervals import *
from merit import solve_problem, solve_with_monotonicity
from helpers import swiss_nsf, top_k
import time 
import json
import pickle     

def run_and_save_results(x, intervals, k, filename, run_monotonicity=False):
    """
    Run the cutting plane algorithm and save results to a file.
    """
    p_opt, v_opt, info = solve_problem(intervals, k, verbose=True)
    if run_monotonicity:
        p_seq, v_seq, info_seq = solve_with_monotonicity(intervals, k, verbose=False)
    p_swiss_seq = [swiss_nsf(intervals, x, i) for i in range(1,k+1)]
    p_top_k = top_k(x, k)

    # Save results
    with open(filename, 'wb') as f:
        results = {
            'p_opt': p_opt,
            'v_opt': v_opt,
            'info': info,
            'p_swiss': p_swiss_seq,
            'top_k': p_top_k
        }
        if run_monotonicity:
            results.update({
            'p_seq': p_seq,
            'v_seq': v_seq,
            'info_seq': info_seq
            })
        with open(filename, 'wb') as pickle_file:
            pickle.dump(results, pickle_file)

def run_computation_test_and_save_results(x, k, intervals, filename):
    """
    Run the cutting plane algorithm and save results to a file.
    """

    n = len(intervals)
    ks = sorted([n // 100, n // 50, n // 20, n // 10, n // 5, n//3, n // 2] + [k])
    res = {}

    for k in ks:
        print(f'Running computation test for k={k}...')
        # Solve with all optimizations
        p_opt, v_opt, info = solve_problem(intervals, k, use_symmetry=True, add_monotonicity_constraints=True, init_prune=True)
        res[k] = info 
    # Save results
    with open(filename, 'wb') as f:
       pickle.dump(res, f)

def run_case_studies(computation_test=False, load_data_only=False, iter=None):
    loaded_data = {}

    def append_iter_to_filename(filename):
        return filename.replace('.pkl', f'_{iter}.pkl') if iter is not None else filename

    print('===========Running Swiss NSF===========')
    t = time.time()
    # Load Swiss NSF data
    x, intervals, intervals90, half_intervals, half_intervals90 = load_swiss_nsf()
    k = 106
    if computation_test:
        run_computation_test_and_save_results(x, k, intervals, append_iter_to_filename('res/swiss_nsf_computation_results.pkl'))
    elif not load_data_only:
        run_and_save_results(x, intervals, k, append_iter_to_filename('res/swiss_nsf_results.pkl'))
    print('Completed Swiss NSF in: ', time.time() - t, ' seconds')
    loaded_data['swiss_nsf'] = {
        'x': x,
        'intervals': intervals,
        'k': k,
        'decisions': None
    }

    print('===========Running Swiss NSF Manski===========')
    t = time.time()
    # Load Swiss NSF data
    x, intervals = load_swiss_manski()
    k = 106
    if computation_test:
        run_computation_test_and_save_results(x, k, intervals, append_iter_to_filename('res/swiss_nsf_manski_computation_results.pkl'))
    elif not load_data_only:
        run_and_save_results(x, intervals, k, append_iter_to_filename('res/swiss_nsf_manski_results.pkl'))
    print('Completed Swiss NSF Manski in: ', time.time() - t, ' seconds')
    loaded_data['swiss_nsf_manski'] = {
        'x': x,
        'intervals': intervals,
        'k': k,
        'decisions': None
    }

    print('===========Running NeurIPS LOO===========')
    t = time.time()
    # Load NeurIPS data
    x, intervals, decisions = load_neurips_leaveoneout()
    k = min(decisions.value_counts())  # number of reject decisions
    if computation_test:
        run_computation_test_and_save_results(x, k, intervals, append_iter_to_filename('res/neuripsloo_computation_results.pkl'))
    elif not load_data_only:
        run_and_save_results(x, intervals, k, append_iter_to_filename('res/neuripsloo_results.pkl'))
    print('Completed Neurips LOO in: ', time.time() - t, ' seconds')
    loaded_data['neuripsloo'] = {
        'x': x,
        'intervals': intervals,
        'k': k,
        'decisions': decisions
    }

    print('===========Running NeurIPS Gaussian===========')
    t = time.time()
    # Load NeurIPS data
    x, intervals50, intervals95, decisions = load_neurips_gaussian_model()
    k = min(decisions.value_counts())  # number of reject decisions
    if computation_test:
        run_computation_test_and_save_results(x, k, intervals50, append_iter_to_filename('res/neuripsgaussian_computation_results.pkl'))
    elif not load_data_only:
        run_and_save_results(x, intervals50, k, append_iter_to_filename('res/neuripsgaussian_results.pkl'))
    print('Completed Neurips Gaussian in: ', time.time() - t, ' seconds')
    loaded_data['neuripsgaussian'] = {
        'x': x,
        'intervals': intervals50,
        'k': k,
        'decisions': decisions
    }

    print('===========Running NeurIPS Subjectivity===========')
    t = time.time()
    # Load NeurIPS data
    x, intervals, decisions = load_neurips_subjectivity_intervals()
    k = min(decisions.value_counts())  # number of reject decisions
    if computation_test:
        run_computation_test_and_save_results(x, k, intervals, append_iter_to_filename('res/neuripssubjectivity_computation_results.pkl'))
    elif not load_data_only:
        run_and_save_results(x, intervals, k, append_iter_to_filename('res/neuripssubjectivity_results.pkl'))
    print('Completed Neurips Subjectivity in: ', time.time() - t, ' seconds')
    loaded_data['neuripssubjectivity'] = {
        'x': x,
        'intervals': intervals,
        'k': k,
        'decisions': decisions
    }

    print('===========Running ICLR LOO===========')
    t = time.time()
    # Load ICLR data
    x, intervals, decisions = load_iclr_leaveoneout()
    k = min(decisions.value_counts())  # number of reject decisions
    if computation_test:
        run_computation_test_and_save_results(x, k, intervals, append_iter_to_filename('res/iclrloo_computation_results.pkl'))
    elif not load_data_only:
        run_and_save_results(x, intervals, k, append_iter_to_filename('res/iclrloo_results.pkl'), run_monotonicity=False)
    print('Completed ICLR LOO in: ', time.time() - t, ' seconds')
    loaded_data['iclrloo'] = {
        'x': x,
        'intervals': intervals,
        'k': k,
        'decisions': decisions
    }

    print('===========Running ICLR Gaussian===========')
    t = time.time()
    # Load ICLR data
    x, intervals50, intervals95, decisions = load_iclr_gaussian_model()
    k = min(decisions.value_counts())  # number of reject decisions
    if computation_test:
        run_computation_test_and_save_results(x, k, intervals50, append_iter_to_filename('res/iclrgaussian_computation_results.pkl'))
    elif not load_data_only:
        run_and_save_results(x, intervals50, k, append_iter_to_filename('res/iclrgaussian_results.pkl'), run_monotonicity=False)
    print('Completed ICLR Gaussian in: ', time.time() - t, ' seconds')
    loaded_data['iclrgaussian'] = {
        'x': x,
        'intervals': intervals50,
        'k': k,
        'decisions': decisions
    }

    print('===========Running ICLR Subjectivity===========')
    t = time.time()
    # Load ICLR data
    x, intervals, decisions = load_iclr_subjectivity_intervals()
    k = min(decisions.value_counts())  # number of reject decisions
    if computation_test:
        run_computation_test_and_save_results(x, k, intervals, append_iter_to_filename('res/iclrsubjectivity_computation_results.pkl'))
    elif not load_data_only:
        run_and_save_results(x, intervals, k, append_iter_to_filename('res/iclrsubjectivity_results.pkl'), run_monotonicity=False)
    print('Completed ICLR Subjectivity in: ', time.time() - t, ' seconds')
    loaded_data['iclrsubjectivity'] = {
        'x': x,
        'intervals': intervals,
        'k': k,
        'decisions': decisions
    }

    return loaded_data


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
    for i in range(10):
        print(f'===========Running Iteration {i} for Computation Tests===========')
        run_case_studies(computation_test=True, iter=i)

    # # print('===========Running Ablations===========')
    # run_data_ablations('neurips')