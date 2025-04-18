import numpy as np
import pandas as pd


###########################################################################
######                   Load Data-Driven Intervals                  ######
###########################################################################

def load_swiss_nsf(PATH='SwissNSFData/intervals.csv'):
    df = pd.read_csv(PATH)
    df.sort_values(by='m', ascending=True, inplace=True) # lower is better for Swiss NSF (ranked by expectd rank)
    df.reset_index(drop=True, inplace=True)
    n = df.shape[0]
    intervals = list(zip(n - df['h'], n - df['l'])) # make list of intervals, l = n - ER, h = n ER using using 50% CI
    intervals90 = list(zip(n - df['hh'], n - df['ll'])) # make list of intervals, l = n - ER, h = n ER using 90% CI
    x = list(n - df['m'])
    half_intervals = [(x[i] - (x[i]-l)/2, x[i] + (h - x[i]) / 2) for i, (l,h) in enumerate(intervals)]
    half_intervals90 = [(x[i] - (x[i]-l)/2, x[i] + (h - x[i]) / 2) for i, (l,h) in enumerate(intervals90)]

    return x, intervals, intervals90, half_intervals, half_intervals90

def load_neurips_leaveoneout(PATH='ConferenceReviewData/neurips2024_data/neurips2024_reviews.csv'):
    df = pd.read_csv(PATH)
    df = df[df.decision != 'Reject'].reset_index(drop=True)
    ratings = df.groupby('paper_id').agg({'rating': list, 'decision': 'max'}).reset_index()
    ratings['decision'] = ratings['decision'].replace({'Accept (poster)': 'Poster', 'Accept (oral)': 'Spotlight/Oral', 'Accept (spotlight)': 'Spotlight/Oral'})

    def get_interval(lst):
        means = []
        for i in range(len(lst)):
            temp_lst = lst[:i] + lst[i+1:]
            mean = 1. * sum(temp_lst) / len(temp_lst)
            means.append(mean)
        return min(means), max(means)
    
    intervals = ratings['rating'].apply(get_interval)
    x = ratings['rating'].apply(np.mean)
    decision = ratings['decision']

    return x, intervals, decision

def load_neurips_minmax(PATH='ConferenceReviewData/neurips2024_data/neurips2024_reviews.csv'):
    df = pd.read_csv(PATH)
    df = df[df.decision != 'Reject'].reset_index(drop=True)
    ratings = df.groupby('paper_id').agg({'rating': ['max', 'mean', 'min', 'count'], 'decision': 'max'}).reset_index()
    ratings.columns = ['paper_id', 'rating_max', 'rating_mean', 'rating_min', 'num_reviews', 'decision']
    # rename Accept (poster) to Poster and Accept (oral) or Accept (spotlight) to Spotlight/Oral
    ratings['decision'] = ratings['decision'].replace({'Accept (poster)': 'Poster', 'Accept (oral)': 'Spotlight/Oral', 'Accept (spotlight)': 'Spotlight/Oral'})
    intervals = list(zip(ratings['rating_min'], ratings['rating_max']))
    x = list(ratings['rating_mean'])
    decisions = ratings['decision']

    return x, intervals, decisions

###########################################################################
######                 Generate Random Intervals                     ######
###########################################################################

# generate n random intervals with endpoints sampled uniformly from [0, M]
def generate_uniform_intervals(n, M=10):
    intervals = []
    for _ in range(n):
        a = np.round(M*np.random.rand(), 5)
        b = np.round(M*np.random.rand(), 5)
        if a > b:
            a, b = b, a
        intervals.append((a,b))
    return intervals

# generate n random intervals by sampling n values of sigma Unif(0, m), n values of mu from N(0, sigma_i) and constructing 95% CI around each x_i
def generate_gaussian_intervals(n, m = 3):
    intervals = []
    for _ in range(n):
        sigma = m*np.random.rand()
        mu = np.random.normal(0, sigma)
        a = np.round(mu - 1.96*sigma, 5)
        b = np.round(mu + 1.96*sigma, 5)
        intervals.append((a,b))
    return intervals