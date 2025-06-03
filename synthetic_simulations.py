import numpy as np 
import pandas as pd
import GPy
import time 
from scipy.stats import pareto

from cutting_plane import solve_problem
from helpers import swiss_nsf, top_k
import os

# Generate random assignment with n items, n reviewers and r reviews per item
def generate_random_assignment(n_items, n_reviewers, items_per_rev):
    A = np.zeros((n_items, n_reviewers), dtype=int)

    item_pool = set(np.arange(n_items))
    item_counter = [0 for i in range(n_items)]

    for i in range(n_reviewers):
        if len(item_pool) < items_per_rev:
            assignment = list(item_pool) + list(np.random.choice(n_items, items_per_rev - len(item_pool), replace=False))
        else:
            assignment = np.random.choice(tuple(item_pool), items_per_rev, replace=False)
        
        for idx in assignment:
            item_counter[idx] += 1
            if item_counter[idx] == items_per_rev*n_reviewers//n_items: ## paper_per
                item_pool.remove(idx)
        A[assignment, i] = 1

    return A

def generate_gaussian_miscalibration_data(
    n_items,
    n_reviewers,
    items_per_rev,
    sigma_theta,
    sigma_b,
    sigma_err,
    sigma_a=None, # if None, no multiplicative miscalibration
    score_range=(1, 10),
):
    """
    Synthetic data generator for the model
        y_pr = a_r(theta_p + e_pr) + b_r

    Returns
    -------
    assignment  : (n_items, n_reviewers) binary matrix
    true_theta  : (n_items,)  ground-truth qualities
    review_scores : (n_items, n_reviewers) matrix with NaNs for unassigned pairs
    """
    # sample true qualities
    theta = np.random.normal(
            loc=(score_range[1] + score_range[0]) / 2.0, scale=sigma_theta, size=(n_items,1)
        )

    # sample linear miscalibration offsets
    b = np.random.normal(0.0, sigma_b, size=n_reviewers)
    if sigma_a is None:
        a = np.ones(n_reviewers)  # no multiplicative miscalibration
    else:
        a = np.random.lognormal(0.0, sigma_a, size=n_reviewers)

    assignment = generate_random_assignment(n_items, n_reviewers, items_per_rev)

    err = np.random.normal(0.0, sigma_err, size=assignment.shape)
    scores = a * (theta + err) + b  # shape (n_items, n_reviewers)
    scores = np.clip(np.round(scores), *score_range)  

    y = np.full_like(scores, np.nan)
    y[assignment == 1] = scores[assignment == 1]  

    return assignment, theta.squeeze(), y

def generate_arbitrary_miscalibration_data(
    n_items,
    n_reviewers,
    revs_per_item,
    sigma_theta,
    sigma_err,
    pct_arbitrary=1.,
    sigma_b=None,
    n_threshold=5,
    score_range=(1, 10),
):
    # sample true qualities
    theta = np.random.normal(
            loc=(score_range[1] + score_range[0]) / 2.0, scale=sigma_theta, size=(n_items,1)
        )
    
    # sample linear miscalibration offsets
    assignment = generate_random_assignment(n_items, n_reviewers, revs_per_item)
    # sample random errors 
    err = np.random.normal(0.0, sigma_err, size=assignment.shape)

    # for each reviewer generate a random monotonic mapping from theta value to scores
    reviewer_mappings = {}
    for r in range(n_reviewers):
        if sigma_b is not None:
            b = np.random.normal(0, sigma_b, 1)
        else:
            b = 0
        if r < pct_arbitrary*n_reviewers:
            thresholds = np.sort(np.random.uniform(min(theta), max(theta), n_threshold))
            scores_for_mapping = np.linspace(score_range[0], score_range[1], n_threshold + 1)
            # function maps score to threshold index
            func = lambda x: scores_for_mapping[np.searchsorted(thresholds, x+b, side='right')]
        else:
            func = lambda x: x+b
        reviewer_mappings[r] = func 
    
    # generate scores by applying the mapping to the true qualities and adding errors
    scores = np.full_like(assignment, np.nan, dtype=float)
    for r in range(n_reviewers):
        # apply the mapping to the true qualities
        mapped_scores = reviewer_mappings[r](theta).squeeze() + err[:, r]
        # clip scores to the range
        mapped_scores = np.clip(np.round(mapped_scores), *score_range)
        # assign scores to the corresponding reviewers
        scores[assignment[:, r] == 1, r] = mapped_scores[assignment[:, r] == 1]
    y = np.full_like(scores, np.nan)
    y[assignment == 1] = scores[assignment == 1]  # assign scores to the corresponding pairs

    return assignment, theta.squeeze(), y

def generate_pareto_miscalibration_data(
    n_items,
    n_reviewers,
    items_per_rev,
    pareto_shape,
    miscal_range,
    sigma_err,
    score_range=(1, 10)
):
    """
    Generate review data with reviewer-specific Pareto-based miscalibration.

    Each reviewer has a quantizes based on the CDF of a Pareto distribution
    with shape parameter sampled around `pareto_shape`.
    """

    # Sample true item quality
    theta = np.random.pareto(pareto_shape, size=n_items) + 1  # add 1 to set xm = 1

    # Reviewer-specific Pareto miscalibration scales (shape params)
    reviewer_shapes = np.random.uniform(
        pareto_shape - miscal_range / 2.0,
        pareto_shape + miscal_range / 2.0,
        size=n_reviewers
    )

    # Generate assignment
    assignment = generate_random_assignment(n_items, n_reviewers, items_per_rev)

    # Generate scores
    scores = np.full((n_items, n_reviewers), np.nan)
   
    for r in range(n_reviewers):
        alpha_r = reviewer_shapes[r]

        reviewer_items = np.where(assignment[:, r] == 1)[0]
        for i in reviewer_items:
            val = theta[i] + np.random.normal(0.0, sigma_err)
            percentile = pareto.cdf(val, alpha_r)
            # Convert percentile to score in 1 to 10 range
            score = (score_range[0] + percentile * (score_range[1] - score_range[0]))
            score += np.random.normal(0.0, sigma_err)  # add noise
            score = np.clip(score, *score_range)
            scores[i, r] = score

    return assignment, theta, scores

def fit_linear_miscalibration_model(A, y):
    ### Bayesian model from Ge et al.

    # Prepare data 
    item_idx_np, reviewer_idx_np = np.nonzero(A)
    y_obs = y[item_idx_np, reviewer_idx_np]

    # Normalize
    mu = np.nanmean(y_obs)
    y_target = y_obs - mu

    # One-hot encode items
    X1 = pd.get_dummies(item_idx_np)
    X1 = X1[sorted(X1.columns, key=int)]

    # One-hot encode reviewers
    X2 = pd.get_dummies(reviewer_idx_np)
    X2 = X2[sorted(X2.columns, key=int)]

    # Concatenate features
    X = X1.join(X2, lsuffix='_item', rsuffix='_rev')

    # Estimate the parameters 
    kern1 = GPy.kern.Linear(input_dim=len(X1.columns), active_dims=np.arange(len(X1.columns)))
    kern1.name = 'K_f'
    kern2 = GPy.kern.Linear(input_dim=len(X2.columns), active_dims=np.arange(len(X1.columns), len(X.columns)))
    kern2.name = 'K_b'

    model = GPy.models.GPRegression(X, y_target[:, None], kern1+kern2)
    model.optimize()

    alpha_f = model.sum.K_f.variances
    alpha_b = model.sum.K_b.variances/alpha_f
    sigma2 = model.Gaussian_noise.variance/alpha_f

    est_params = {
            'sigma_theta': np.sqrt(alpha_f),
            'sigma_b': np.sqrt(model.sum.K_b.variances),
            'sigma_err': np.sqrt(model.Gaussian_noise.variance)}

    K_f = np.dot(X1, X1.T)
    K_b = alpha_b*np.dot(X2, X2.T)
    K = K_f + K_b + sigma2*np.eye(X2.shape[0])
    Kinv, _, _, _ = GPy.util.linalg.pdinv(K) # since we have GPy loaded in use their positive definite inverse.
    alpha = np.dot(Kinv, y_target)
    yTKinvy = np.dot(y_target, alpha)
    alpha_f = yTKinvy/len(y_target)

    K_s = K_f + np.eye(K_f.shape[0])*sigma2
    means = pd.Series(np.dot(K_s, alpha) + mu, index=X1.index)
    covs = alpha_f*(K_s - np.dot(K_s, np.dot(Kinv, K_s)))

    scores = np.random.multivariate_normal(mean=means, cov=covs, size=5000).T
    item_score = pd.DataFrame(np.dot(np.diag(1./X1.sum(0)), np.dot(X1.T, scores)), index=X1.columns)
    item_rank = item_score.rank(axis=0, method='first', ascending=True)

    expected_rank = item_rank.mean(axis=1)
    intervals50 =  item_rank.quantile([0.25, 0.75], axis=1).to_numpy().T
    intervals95 = item_rank.quantile([0.025, 0.975], axis=1).to_numpy().T
    
    return expected_rank, intervals50, intervals95, est_params


def run_miscalibration_simulation(n_items, n_reviewers, items_per_rev, error_params, error_type, ks, n_trials=10):
    results = []
    for iter in range(n_trials):
        print(f"Running iteration {iter + 1}/{n_trials}...")

        # Generate data
        if error_type == 'linear':
            sigma_theta=error_params['sigma_theta']
            sigma_b=error_params['sigma_b']
            sigma_err=error_params['sigma_err']
            sigma_a=error_params['sigma_a']
            A, theta, y = generate_gaussian_miscalibration_data(
                n_items, n_reviewers, items_per_rev, sigma_theta, sigma_b, sigma_err, sigma_a)
        elif error_type == 'arbitrary':
            sigma_theta=error_params['sigma_theta']
            sigma_err=error_params['sigma_err']
            pct_arbitrary=error_params['pct_arbitrary']
            sigma_b=error_params['sigma_b']
            n_threshold=error_params['n_threshold']

            A, theta, y = generate_arbitrary_miscalibration_data(
                n_items, n_reviewers, items_per_rev, sigma_theta, sigma_err, pct_arbitrary, sigma_b, n_threshold)
        elif error_type == 'pareto':
            pareto_shape = error_params['pareto_shape']
            miscal_range = error_params['miscal_range']
            sigma_err = error_params['sigma_err']
            A, theta, y = generate_pareto_miscalibration_data(
                n_items, n_reviewers, items_per_rev, pareto_shape, miscal_range, sigma_err)
        else:
            raise(Exception('Miscalibration type must be linear, arbitrary, or riskybias.'))

        er, intervals50, intervals95, est_params = fit_linear_miscalibration_model(A, y)
        raw_means = np.nanmean(y, axis=1)
        raw_medians = np.nanmedian(y, axis=1)

        # calculate typical width of intervals 50
        intervals50_width = np.mean(intervals50[:, 1] - intervals50[:, 0])

        for k in ks:
            true_top_k = np.argsort(-1*(theta))[:k]
            total_quality = theta[true_top_k].sum()
            p_top_k = top_k(theta, k)

            p_er_deterministic = top_k(er, k)
            p_raw_deterministic = top_k(raw_means, k)
            p_raw_deterministic_median = top_k(raw_medians, k)
            p_merit,_,_ = solve_problem(intervals50, k)
            p_swiss = np.array(swiss_nsf(intervals50, er, k))
            
            # get % total quality of each method
            p_quality_er_deterministic = np.dot(theta, p_er_deterministic) / total_quality
            p_quality_raw_deterministic = np.dot(theta, p_raw_deterministic) / total_quality
            p_quality_raw_deterministic_median = np.dot(theta, p_raw_deterministic_median) / total_quality
            p_quality_merit = np.dot(theta, p_merit) / total_quality
            p_quality_swiss = np.dot(theta, p_swiss) / total_quality
            
            # get % of true top k items selected
            prec_er_deterministic = np.dot(p_top_k, p_er_deterministic) / k
            prec_raw_deterministic = np.dot(p_top_k, p_raw_deterministic) / k
            prec_raw_deterministic_median = np.dot(p_top_k, p_raw_deterministic_median) / k
            prec_merit = np.dot(p_top_k, p_merit) / k
            prec_swiss = np.dot(p_top_k, p_swiss) / k

            # save % randomized (p > 0 and p < 1)
            n_rand_swiss = np.sum((p_swiss > 0) & (p_swiss < 1))
            n_rand_merit = np.sum((p_merit > 0) & (p_merit < 1))

            if prec_merit - prec_swiss >= 0.1:
                print(f"Interesting: Merit and Swiss methods differ by more than 0.1 in precision for k={k} in iteration {iter + 1}")
                # save intervals and probabilities to a file
                os.makedirs('res/simulation_results/interesting_cases', exist_ok=True)
                np.savez(
                    f'res/simulation_results/interesting_cases/iter{iter+1}_k{k}_sigma_b{error_params.get("sigma_b", "NA")}.npz',
                    intervals50=intervals50,
                    p_merit=p_merit,
                    p_swiss=p_swiss,
                    er=er,
                    theta=theta,
                    y=y,
                    A=A
                )

            results.append([
                iter, n_reviewers, n_items, items_per_rev, error_type,
                k, intervals50_width, est_params,
                n_rand_swiss, n_rand_merit,
                # p_top_k, p_er_deterministic, p_raw_deterministic, p_raw_deterministic_median, p_merit, p_swiss,
                p_quality_er_deterministic, p_quality_raw_deterministic, p_quality_raw_deterministic_median, p_quality_merit, p_quality_swiss, 
                prec_er_deterministic, prec_raw_deterministic, prec_raw_deterministic_median, prec_merit, prec_swiss
            ])

    d = pd.DataFrame(results, columns=[
        'iter', 'n_reviewers', 'n_items', 'items_per_rev', 'error_type',
        'k', 'er_interval_mean_width', 'estimated_params',
        'n_rand_swiss', 'n_rand_merit',
        # 'p_top_k', 'p_er_deterministic', 'p_raw_deterministic', 'p_raw_deterministic_median', 'p_merit', 'p_swiss',
        'prec_quality_er_deterministic', 'prec_quality_raw_deterministic', 'prec_quality_raw_deterministic_median', 'prec_quality_merit', 'prec_quality_swiss',
        'prec_er_deterministic', 'prec_raw_deterministic', 'prec_raw_deterministic_median', 'prec_merit', 'prec_swiss'
    ])

    # add error_params to DataFrame
    for key, value in error_params.items():
        d[key] = value
    return d

def simulate_risky_bias(n_reviewers, n_items, items_per_rev, k, model_params, score_range=(1,10)):
    alpha = model_params['alpha']
    p_bias = model_params['p_bias']
    sigma_err = model_params['sigma_err']

    A = generate_random_assignment(n_items, n_reviewers, items_per_rev)
    theta = pareto.rvs(alpha, size=n_items)
    risky_bias = np.random.choice([0, 1], size=n_reviewers, p=[1-p_bias, p_bias])

    # Broadcast theta[i] to assigned (i, j) pairs
    item_idx, reviewer_idx = np.where(A == 1)
    n_obs = len(item_idx)

    get_score = lambda t, alpha: score_range[0] + (score_range[1] - score_range[0])*pareto.cdf(t, alpha) 

    # Generate raw scores
    raw_scores = np.array([get_score(theta[i], alpha) for i in item_idx])

    # Apply risky bias: reviewers with bias & score above threshold get middle score
    biased = risky_bias[reviewer_idx] == 1
    bias_mask = biased & (raw_scores > score_range[1] - 1)
    raw_scores[bias_mask] = score_range[0] + (score_range[1] - score_range[0]) / 2.

    # Add noise and clip/round
    raw_scores += np.random.normal(0, sigma_err, size=n_obs)
    raw_scores = np.clip(np.round(raw_scores), score_range[0], score_range[1])

    # Fill into score matrix
    y = np.full((n_items, n_reviewers), np.nan)
    y[item_idx, reviewer_idx] = raw_scores

    # Generate intervals and point estimates
    lower_bounds = np.nanmin(y, axis=1)
    upper_bounds = np.nanmax(y, axis=1)
    x_mean = np.nanmean(y, axis=1)
    x_median = np.nanmedian(y, axis=1)
    intervals = list(zip(lower_bounds, upper_bounds))

    # Evaluate selection policies
    p_merit, _, _ = solve_problem(intervals, k)
    p_swiss = swiss_nsf(intervals, x_mean, k)
    p_deterministic_mean = top_k(x_mean, k)
    p_deterministic_median = top_k(x_median, k)

    best_quality = np.dot(top_k(theta, k), theta)
    q_merit = np.dot(p_merit, theta) / best_quality
    q_swiss = np.dot(p_swiss, theta) / best_quality
    q_deterministic_mean = np.dot(p_deterministic_mean, theta) / best_quality
    q_deterministic_median = np.dot(p_deterministic_median, theta) / best_quality

    return q_merit, q_swiss, q_deterministic_mean, q_deterministic_median


CONFERENCE_PARAMS = {
    'n_reviewers': 1000,
    'n_items': 1000,
    'items_per_rev': 5,
    'score_range': (1, 10),
    }

SWISS_NSF_PARAMS = {
    'n_reviewers': 10,
    'n_items': 350,
    'items_per_rev': 80,
    'score_range': (1, 10),
}    

def run_linear_miscalibration_sims(vary_param, param_values, PARAMS=SWISS_NSF_PARAMS):
    linear_miscalibration_params={
        'sigma_theta': 2.0,
        'sigma_b': 2.0,
        'sigma_err': 0.5,
        'sigma_a': None
    }

    # Swiss NSF parameters
    n_reviewers = SWISS_NSF_PARAMS['n_reviewers']
    n_items = SWISS_NSF_PARAMS['n_items']
    items_per_rev = SWISS_NSF_PARAMS['items_per_rev']
    score_range = SWISS_NSF_PARAMS['score_range']
    n_trials = 50
    ks = [n_items // 10, n_items // 3]

    print(f'Running simulations with Swiss NSF parameters')
    linear_results = []
    for param in param_values:
        linear_miscalibration_params[vary_param] = param
        print(f'Running simulations with {vary_param}={param}')
        t = time.time()
        
        results = run_miscalibration_simulation(
            n_reviewers=n_reviewers,
            n_items=n_items,
            items_per_rev=items_per_rev,
            error_params=linear_miscalibration_params,
            error_type='linear',
            n_trials=n_trials,
            ks=ks,
        )

        print(f'Simulation with {vary_param}={param} took {time.time() - t:.2f} seconds')
        
        linear_results.append(results)
    # Convert results to DataFrame
    linear_results_df = pd.concat(linear_results)
    return linear_results_df

def run_riskybias_sims(vary_param, param_values, PARAMS=SWISS_NSF_PARAMS, n_trials=1000):
    n_reviewers = PARAMS['n_reviewers']
    n_items = PARAMS['n_items']
    items_per_rev = PARAMS['items_per_rev']
    score_range = PARAMS['score_range']
    n_trials = 1000

    k = 1 + (n_items // (score_range[1] - score_range[0]))

    error_params = {
        'sigma_err': 0.5,
        'p_bias': 0.5,
        'alpha': 1.5
    }
    
    res = {}
    for param in param_values:
        res[param] = []
        # Run simulations for each p_bias
        print(f'Simulating with param {vary_param}:', param)
        error_params[vary_param] = param
        t = time.time()
        for sim in range(n_trials):
            q_merit, q_swiss, q_deterministic_mean, q_deterministic_median = simulate_risky_bias(
                n_reviewers, n_items, items_per_rev, k, error_params, score_range
            )
            res[param].append({
                    'q_merit': q_merit,
                    'q_swiss': q_swiss,
                    'q_deterministic_mean': q_deterministic_mean,
                    'q_deterministic_median': q_deterministic_median
                })
        print('Simulation time:', time.time() - t, 'seconds')

    # make res into a DataFrame
    results = []
    for param, values in res.items():
        for value in values:
            results.append({
                vary_param: param,
                'q_merit': value['q_merit'],
                'q_swiss': value['q_swiss'],
                'q_deterministic_mean': value['q_deterministic_mean'],
                'q_deterministic_median': value['q_deterministic_median']
            })
    df_results = pd.DataFrame(results)

    return df_results

if __name__ == "__main__":
   ## Run linear miscalibration sims
    df = run_linear_miscalibration_sims('sigma_b', [0.0, 0.5, 1.0, 2.0, 4.0], PARAMS=SWISS_NSF_PARAMS)
    df.to_csv('res/simulation_results/linear_miscalibration_results_swissnsfparams.csv', index=False)

    df = run_linear_miscalibration_sims('sigma_a', [0.0, 0.5, 1.0, 2.0, 4.0], PARAMS=SWISS_NSF_PARAMS)
    df.to_csv('res/simulation_results/linear_miscalibration_results_swissnsfparams_misspecified_sigmaa.csv', index=False)

    df = run_linear_miscalibration_sims('sigma_b', [0.0, 0.5, 1.0, 2.0, 4.0], PARAMS=CONFERENCE_PARAMS)
    df.to_csv('res/simulation_results/linear_miscalibration_results_conferenceparams.csv', index=False)

    df = run_linear_miscalibration_sims('sigma_a', [0.0, 0.5, 1.0, 2.0, 4.0], PARAMS=CONFERENCE_PARAMS)
    df.to_csv('res/simulation_results/linear_miscalibration_results_conferenceparams_misspecified_sigmaa.csv', index=False)


    ### Run simulations for risky bias
    df = run_riskybias_sims(vary_param='p_bias', param_values=np.arange(0.0, 1.0, 0.1), PARAMS=SWISS_NSF_PARAMS)
    df.to_csv('res/simulation_results/riskybias_swissnsf_pbias.csv', index=False)

    df = run_riskybias_sims(vary_param='sigma_err', param_values=[0.0, 0.5, 1.0, 2.0, 4.0], PARAMS=SWISS_NSF_PARAMS)
    df.to_csv('res/simulation_results/riskybias_swissnsf_sigma_err.csv', index=False)

    df = run_riskybias_sims(vary_param='alpha', param_values=[1.0, 1.25, 1.5, 2.0, 4.0],PARAMS=SWISS_NSF_PARAMS)
    df.to_csv('res/simulation_results/riskybias_swissnsf_alpha.csv', index=False)


    