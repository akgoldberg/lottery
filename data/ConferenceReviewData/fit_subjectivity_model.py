import csv
import numpy as np
import cvxpy as cvx
import copy
np.set_printoptions(suppress=True)
SMALL_NUM = 0.0001 #Used in the optimization to break ties

def fit_subjectivity_model(df, CRITERIA=['soundness', 'presentation', 'contribution'], OVERALL='rating'):
    overall_scores = df[OVERALL].astype(float) #Overall scores are stored in this array
    criteria_scores = df[CRITERIA].astype(float) #Criteria scores are stored in this array
    num_criteria = len(CRITERIA) #Number of criteria
    num_samples = len(criteria_scores) #Total number of reviews

    #Run algorithm
    unique_criteria_scores, inverse_indices = np.unique(criteria_scores,axis=0,return_inverse=True)

    f = cvx.Variable(len(unique_criteria_scores))	# we have a variable for f-value on each of the x-values of the data
    loss = 0
    for sample in np.arange(num_samples):
        loss += cvx.pnorm(overall_scores[sample] - f[inverse_indices[sample]] ,1)+  SMALL_NUM*cvx.pnorm(overall_scores[sample] - f[inverse_indices[sample]] ,1)**2 #the first term is the L1 norm and the second one is for tie breaking -- a small value times the the squared difference
    
    obj = cvx.Minimize(loss)

    constraints = []
    # Adding all pairwise monotonicity constraints
    for sample1 in np.arange(len(unique_criteria_scores)):
        for sample2 in np.arange(len(unique_criteria_scores)):
            if(sum(unique_criteria_scores[sample1,:] >= unique_criteria_scores[sample2,:]) == 2*num_criteria):#We have 2*num_criteria since the number of criteria has doubled due to addition of the optional_flag columns
                constraints.append(f[sample1] >= f[sample2])
            if(sum(unique_criteria_scores[sample2,:] >= unique_criteria_scores[sample1,:]) == 2*num_criteria):
                constraints.append(f[sample2] >= f[sample1])

    prob = cvx.Problem(obj, constraints)
    prob.solve(verbose=True,max_iter=50000)

    adjusted_scores = np.zeros(num_samples)

    for sampleindex in np.arange(num_samples):
        adjusted_scores[sampleindex] = np.round( f.value[inverse_indices[sampleindex]],  4)

    return adjusted_scores