.. _dev_installation:

Installation
============

Obtain the source code from GitHub at `grimme-lab/dxtb <https://github.com/grimme-lab/dxtb>`__ by cloning the repository with

.. code::

    git clone https://github.com/grimme-lab/dxtb
    cd dxtb

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code::

    mamba env create -n torch -f environment.yaml
    mamba activate torch

For development, additionally install the development tools in your environment, and use the ``-e`` option to install the package in editable mode.

.. code::

   pip install -e .[dev]

The pre-commit hooks are initialized by running the following command in the root of the repository.

.. code::

   pre-commit install
