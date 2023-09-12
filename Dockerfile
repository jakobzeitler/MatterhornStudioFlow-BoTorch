FROM mambaorg/micromamba:1.4.9
COPY --chown=$MAMBA_USER:$MAMBA_USER conda.yml /tmp/conda.yml
RUN micromamba install -y -n base -f /tmp/conda.yml \
    && micromamba clean -a -y
USER root
USER root
RUN apt-get update -y && apt-get install -y procps