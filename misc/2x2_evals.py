### 2 proposals, 2 reviewers ###

import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import pandas as pd 
import subprocess

# Sample from the Swiss NSF model
def sample_swiss_model1(sigma_nu=0.5):
    # sample tau_theta uniform at random from [0,2]
    tau_theta = np.random.uniform(0, 2)
    # sample tau_lambda uniform at random from [0,2]
    tau_lambda = np.random.uniform(0, 2)
    # sample sigma uniform at random from [0,2]
    sigma = np.random.uniform(0, 2)
    # sample nu_1, nu_2 iid uniform at random from N(0,0.5^2)
    nu = np.random.normal(0, sigma_nu**2, size=2)
    
    theta = []
    lambda_ = []
    y = []
    
    # for i in 1,2
    for i in range(2):
        # sample theta_i from N(nu_j, tau_theta^2)
        theta_i = np.random.normal(0, tau_theta)
        theta.append(theta_i)
        
        # sample lambda_i from N(nu_j, tau_lambda^2)
        lambda_i = np.random.normal(nu[i], tau_lambda)
        lambda_.append(lambda_i)
        
        # sample y_i from N(theta_i + lambda_i, sigma^2)
        y_i = np.random.normal(theta_i + lambda_i, sigma)
        y.append(y_i)    
    
    return theta, y

### Return probability each method is correct
def deterministic_correct(thetas, ys):
    if thetas[0] > thetas[1]:
        return ys[0] > ys[1]
    else:
        return ys[1] > ys[0]

def threshold_lottery_correct(thetas, ys, C):
    if np.abs(ys[0] - ys[1]) > C:
        return int(deterministic_correct(thetas, ys))
    else:
        return 0.5

# Run the Swiss NSF estimator using R code 
def swiss_estimator_correct(thetas, ys):
    # make data frame with ys as scores
    df = pd.DataFrame({
                        'proposal': range(len(ys)),
                        'assessor': range(len(ys)),
                        'grade_variable': ys}
                    )
    # save data to csv 
    df.to_csv('sample.csv', index=False)
    # run R script run_swiss_nsf.R
    result = subprocess.run(['Rscript', 'run_swiss_nsf.R'], capture_output=True)
    # read in results
    res = result.stdout.decode('utf-8').split('\n')
    # return the result
    return res 
