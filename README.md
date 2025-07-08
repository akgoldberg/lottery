# Randomized Selection under Uncertainty (Peer Review Lottery)

This directory contains code to implement and evaluate methods for randomized selection (peer review lotteries) based on our paper ["A Principled Approach to Randomized Selection under
Uncertainty: Applications to Peer Review and Grant Funding"](https://arxiv.org/pdf/2506.19083).

## Structure

- `merit.py` — Source code implementing the MERIT algorithm
- `dataset_sims.py` - Code to implement experiments using Swiss NSF and conference data with worst-case intervals.
- `synthetic_sims.py` - Code to implement experiments on fully synthetic data under linear miscalibration model.
- `SwissNSFData/` — All code to obtain Swiss NSF dataset
- `ConferenceReviewData/` — All code to obtain conference data (ICLR 2025 and NeurIPS 2024)

## Getting Started

1. Clone the repository.
2. Install python packages (see `requirements.txt`). NOTE: to run experiments you will additionally need to install cvxpy and GPy with pip to generate intervals, but these are not necessary to run MERIT.
3. Install Gurobi: MERIT uses Gurobi to efficiently solve large linear programs. Academics can obtain a license to this software for free. Please follow [instructions from Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/) to install and activate the license.
4. Run example.ipynb notebook to test simple usage of MERIT.
