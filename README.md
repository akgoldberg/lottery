# Randomized Selection under Uncertainty (Peer Review Lottery)

This directory contains code to implement and evaluate methods for randomized selection (peer review lotteries) based on our paper ["A Principled Approach to Randomized Selection under
Uncertainty: Applications to Peer Review and Grant Funding"](https://arxiv.org/pdf/2506.19083).

## Getting Started

1. Clone the repository.
2. Install python packages (run `pip install -r requirements.txt` in terminal).*
3. Install Gurobi: MERIT uses Gurobi to efficiently solve large linear programs. Academics can obtain a license to this software for free. Please follow [instructions from Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/) to install and activate the license.

*Note: to replicate our synthetic data experiments you will additionally need to install the Python pacakges `cvxpy` and `GPy`, but these are not necessary to run MERIT.

### Running the MERIT Selection Algorithm

In order to run MERIT on your own data, follow the example given in `example.ipynb`.

### Replicating Experiments

In order to replicate synthetic data experiments from the paper "A Principled Approach to Randomized Selection under
Uncertainty: Applications to Peer Review and Grant Funding", run python scripts `dataset_sims.py` (for worst-case model) and `synthetic_sims.py` (for probabilistic model.) Then, analysis of the generated data is replicable in the iPython notebooks `dataset_analysis.ipynb` and `synthetic_analysis.ipynb`.

## Directory Structure

- `merit.py` — Source code implementing the MERIT algorithm
- `dataset_sims.py` - Code to implement experiments using Swiss NSF and conference data with worst-case intervals.
- `synthetic_sims.py` - Code to implement experiments on fully synthetic data under linear miscalibration model.
- `data/SwissNSFData/` — All code to obtain Swiss NSF dataset
- `data/ConferenceReviewData/` — All code to obtain conference data (ICLR 2025 and NeurIPS 2024)