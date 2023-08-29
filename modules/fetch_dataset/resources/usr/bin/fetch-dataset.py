#!/usr/bin/env python3

import argparse
import json
import pandas as pd

if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser(description='Download an OpenML dataset')
    parser.add_argument('--token', help='token', required=True)
    parser.add_argument('--project_id', help='project id', required=True)
    parser.add_argument('--data', help='data file', default='data.txt')
    parser.add_argument('--meta', help='metadata file', default='meta.json')

    args = parser.parse_args()

    # 1. Initialise API client
    from MHSapi.MHSapi import MHSapiClient
    client = MHSapiClient(token=args.token, dev=True)
    projects = client.experiments_list()
    project = [p for p in projects if int(p.id) == int(args.project_id)][0]

    # 2. Download dataset
    dataset = client.experiment_data(project)
    print("Data")
    print(dataset)

    # 3. Save data
    dataset.to_csv(args.data, sep='\t')

    # save metadata
    meta = {
        'project_id': args.project_id,
    }

    with open(args.meta, 'w') as f:
        json.dump(meta, f)
