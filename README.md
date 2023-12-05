# OptCore OptApp using BoTorch and SingleTaskGP

A proof-of-concept pipeline for performing hyperparameter optimization of machine learning models with Nextflow.

## Requirements

* Unix-like operating system (Linux, macOS, etc)
* Java >=11
* [Conda](https://docs.conda.io/en/latest/) or [Docker](https://docs.docker.com/)


## Quickstart

1. Install Nextflow (version 22.10.x or higher):
    ```bash
    curl -s https://get.nextflow.io | bash
    ```

2. Launch the pipeline:
    ```bash
    # use conda natively (requires Conda)
    ./nextflow run nextflow-io/hyperopt -profile conda

    # use Wave containers (requires Docker)
    ./nextflow run nextflow-io/hyperopt -profile wave
    ```

3. When the pipeline completes, you can view the training and prediction results in the `results` folder.

Note: the first time you execute the pipeline, Nextflow will take a few minutes to download the pipeline code from this GitHub repository and any related software dependencies (e.g. conda packages or Docker images).

The hyperopt pipeline uses Python (>=3.10) and several Python packages for machine learning and data science. These dependencies are defined in the `conda.yml` file.
