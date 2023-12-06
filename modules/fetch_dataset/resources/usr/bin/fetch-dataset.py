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

    import os
    print(os.environ['CONDA_DEFAULT_ENV'])
    import sys

    print(sys.path)

    # 1. Initialise API client
    import MHSapi
    from MHSapi.MHSapi import MHSapiClient
    from importlib.metadata import version

    client = MHSapi.MHSapi.MHSapiClient(token=args.token, base_url=args.base_url)
    print(version('MHSapi'))
    print(MHSapi.__file__)
    print(dir(MHSapi))
    object_methods = [method_name for method_name in dir(client)
                      if callable(getattr(client, method_name))]
    print(object_methods)

    projects = client.experiments_list()
    project = [p for p in projects if int(p.id) == int(args.project_id)][0]
    parameters = client.parameters_list(project)

    # 2. Download dataset
    dataset = client.experiment_data(project)
    print("Data")
    print(dataset)

    # 3. Get OptApp options

    opt_runs = client.opt_run_list(project)
    print(opt_runs)
    opt_run = [p for p in opt_runs if int(p.id) == int(args.opt_run_id)][0]
    print(opt_run)
    #type = type(opt_run.run_options)
    #method_list = [func for func in dir(type) if callable(getattr(type, func))]
    #print(method_list)
    import json
    run_options = json.loads(opt_run.run_options.json())

    batch_size = 1
    if 'batch_size' in run_options.keys():
        batch_size = run_options['batch_size']

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

    #from botorch.acquisition import UpperConfidenceBound
    #UCB = UpperConfidenceBound(gp, beta=0.1)

    from botorch.optim import optimize_acqf

    from botorch import fit_gpytorch_mll
    from botorch.acquisition.monte_carlo import (
        qExpectedImprovement,
        qNoisyExpectedImprovement,
    )
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.exceptions import BadInitialCandidatesWarning

    upper_bounds = torch.tensor([p.upper_bound for p in inputs])
    lower_bounds = torch.tensor([p.lower_bound for p in inputs])
    bounds = torch.stack([lower_bounds, upper_bounds])
    print(f"batch_size={batch_size}")

    SMOKE_TEST = False
    MC_SAMPLES = 256 if not SMOKE_TEST else 32
    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    qNEI = qNoisyExpectedImprovement(
        model=gp,
        X_baseline=train_X,
        sampler=qmc_sampler,
    )
    candidates, acq_value = optimize_acqf(
        acq_function=qNEI, bounds=bounds, q=int(batch_size), num_restarts=5, raw_samples=20,
    )
    candidates  # tensor([0.4887, 0.5063])
    print("Candidates (raw):")
    print(candidates.detach())

    candidates = pd.DataFrame(candidates.numpy())
    candidates.columns = [input.parameter_text for input in inputs]
    candidates[outcome.parameter_text] = np.nan
    candidates["opt_run_id"] = args.opt_run_id

    print(candidates)
    client.experiment_update_data(project, candidates)