FROM python:3.6-slim
MAINTAINER Dave Cole <dave.cole@data61.csiro.au>

WORKDIR /usr/src/landshark

RUN apt-get update && apt-get upgrade \
    && apt-get install -y --no-install-recommends \
        make \
        gcc \
        libc6-dev \
        libopenblas-base \
        libgdal20 \
        libhdf5-100 \
    && rm -rf /var/lib/apt/lists/* \
    && alias pip=pip3
