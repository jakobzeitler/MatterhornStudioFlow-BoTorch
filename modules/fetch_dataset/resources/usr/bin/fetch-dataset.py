#!/usr/bin/env python3

import argparse
import json

import numpy as np
import pandas as pd

if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Download an OpenML dataset')
    parser.add_argument('--token', help='token', required=True, default='01ba5d69504449ce8de4faca341e8187a29c938bb435aad29b1b8487941ccebf')
    parser.add_argument('--base_url', help='base url', required=True, default= 'https://mhs.ngrok.app/' )
    parser.add_argument('--project_id', help='project id', required=True, default=-1)
    parser.add_argument('--opt_run_id', help='opt run id', required=True, default=-1)
    parser.add_argument('--data', help='data file', default='data.txt')
    parser.add_argument('--meta', help='metadata file', default='meta.json')

    args = parser.parse_args()

    # 1. Initialise API client
    from MHSapi.MHSapi import MHSapiClient
    client = MHSapiClient(token=args.token, base_url=args.base_url)
    projects = client.experiments_list()
    project = [p for p in projects if int(p.id) == int(args.project_id)][0]
    parameters = client.parameters_list(project)

    # 2. Download dataset
    dataset = client.experiment_data(project)
    print("Data")
    print(dataset)

    # 3. Save data
    dataset.to_csv(args.data, sep='\t')


    # 4. Do BO
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.utils import standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize

    inputs = [p for p in parameters if p.outcome == False and p.timestamp == False]
    outcome = [p for p in parameters if p.outcome == True][0]
    X = dataset[[i.parameter_text for i in inputs]]
    Y = dataset[[outcome.parameter_text]]
    train_X = torch.tensor(X.to_numpy(dtype=np.float64))
    train_Y = torch.tensor(Y.to_numpy(dtype=np.float64))
    #train_Y = standardize(train_Y)
    print(train_X.shape)
    print(train_X)
    print(train_Y.shape)
    print(train_Y)

    gp = SingleTaskGP(train_X, train_Y, input_transform=Normalize(d=train_X.shape[-1]), outcome_transform=Standardize(m=train_Y.shape[-1]))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    from botorch.acquisition import UpperConfidenceBound
    UCB = UpperConfidenceBound(gp, beta=0.1)

    from botorch.optim import optimize_acqf

    upper_bounds = torch.tensor([p.upper_bound for p in inputs])
    lower_bounds = torch.tensor([p.lower_bound for p in inputs])
    bounds = torch.stack([lower_bounds, upper_bounds])
    candidate, acq_value = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )
    candidate  # tensor([0.4887, 0.5063])
    print(list(candidate[0]))

    new_sample = {}
    for i, c in enumerate(list(candidate[0])):
        new_sample[inputs[i].parameter_text] = c.numpy()

    new_sample[outcome.parameter_text] = np.nan
    new_sample["opt_run_id"] = args.opt_run_id
    new_sample = pd.DataFrame(new_sample, index=[0])
    print(new_sample)
    client.experiment_update_data(project, new_sample)

    # save metadata
    meta = {
        'project_id': args.project_id,
        'base_url':client.get_base_url()
    }

    with open(args.meta, 'w') as f:
        json.dump(meta, f)
