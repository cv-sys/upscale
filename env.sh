#!/bin/zsh

ENV_NAME="upscale-cvsys"

if conda env list | grep -q "$ENV_NAME"; then
    conda activate "$ENV_NAME"
else
    conda env create -f env.yml
    conda activate "$ENV_NAME"
fi
