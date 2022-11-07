Fully Differentiable Extended Tight Binding
===========================================

.. image:: https://img.shields.io/badge/python-3.10.6-blue.svg
    :target: https://www.python.org/downloads/release/python-3106/

.. image:: https://img.shields.io/badge/PyTorch-1.11-blue.svg
    :target: https://pytorch.org/blog/pytorch-1.11-released/

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://img.shields.io/badge/docstring-numpydoc-black
    :target: https://github.com/psf/black

This project provides a fully differentiable implementation of the extended tight binding (xTB) Hamiltonian.

For details on the xTB methods see

- C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert, S. Spicher, S. Grimme,
  *WIREs Comput. Mol. Sci.*, **2020**, 11, e01493.
  (`DOI <https://doi.org/10.1002/wcms.1493>`__)
- C. Bannwarth, S. Ehlert, S. Grimme,
  *J. Chem. Theory Comput.*, **2019**, 15, 1652-1671.
  (`DOI <https://dx.doi.org/10.1021/acs.jctc.8b01176>`__)
- S. Grimme, C. Bannwarth, P. Shushkov,
  *J. Chem. Theory Comput.*, **2017**, 13, 1989-2009.
  (`DOI <https://dx.doi.org/10.1021/acs.jctc.7b00118>`__)

For alternative implementations also check out

`tblite <https://tblite.readthedocs.io>`__:
  Light-weight tight-binding framework implemented in Fortran with Python bindings


Installation
------------

This project is hosted on GitHub at `grimme-lab/dxtb <https://github.com/grimme-lab/dxtb>`__.
Obtain the source by cloning the repository with

.. code::

   git clone https://github.com/grimme-lab/dxtb
   cd dxtb

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code::

   mamba env create -n torch -f environment.yml
   mamba activate torch

Install this project with pip in the environment

.. code::

   pip install .


The following dependencies are required

- `numpy <https://numpy.org/>`__
- `tomli <https://github.com/hukkin/tomli>`__
- `torch <https://pytorch.org/>`__ version 1.11
- `tad-dftd3 <https://github.com/dftd3/tad-dftd3>`__
- `pydantic <https://github.com/samuelcolvin/pydantic>`__
- `pytest <https://docs.pytest.org/>`__ (tests only)

You can check your installation by running the test suite with

.. code::

   pytest test/ --pyargs dxtb --doctest-modules

or with dxtb module path in pyproject.toml:

.. code::

   python -m pytest test/ --doctest-modules


Development
-----------

For development, additionally install the following tools in your environment.

.. code::

   mamba install black pre-commit pylint pytest pytest-cov


With pip, add the option ``-e`` and the development dependencies for installing in development mode.

.. code::

   pip install -e .[dev]

The pre-commit hooks are initialized by running the following command in the root of the repository.

.. code::

   pre-commit install
