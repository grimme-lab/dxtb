#!/usr/bin/env bash

# Create virtual environments with different versions of PyTorch

env_name="venv-torch"

for version in 1.11.0 1.12.1 1.13.0; do
  if [ ! -d ${env_name}-${version} ]; then
    virtualenv ${env_name}-${version}
    . ${env_name}-${version}/bin/activate
    python -m pip install --upgrade pip
    pip install torch==$version+cpu -f https://download.pytorch.org/whl/torch_stable.html
    pip install -e .[dev]
    deactivate
  fi
done
