FROM mambaorg/micromamba:1.4.1

LABEL image.author.name "Jakob Zeitler"
LABEL image.author.email "jakob@matterhorn.studio"

COPY --chown=$MAMBA_USER:$MAMBA_USER conda.yml /tmp/conda.yml
RUN micromamba install -y -n base -f /tmp/conda.yml && \
    micromamba clean -a -y
USER root
RUN apt-get update -y && apt-get install -y procps