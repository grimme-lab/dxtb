.. _quickstart_installation:

Installation
============

pip
---

.. image:: https://img.shields.io/pypi/v/dxtb
    :target: https://pypi.org/project/dxtb/
    :alt: PyPI Version

*dxtb* can easily be installed with ``pip``.

.. code::

    pip install dxtb


conda
-----

.. image:: https://img.shields.io/conda/vn/conda-forge/dxtb.svg
    :target: https://anaconda.org/conda-forge/dxtb
    :alt: Conda Version

*dxtb* is also available on `conda <https://conda.io/>`__.

.. code::

    mamba install dxtb


From source
-----------

This project is hosted on GitHub at `grimme-lab/dxtb <https://github.com/grimme-lab/dxtb>`__.
Obtain the source by cloning the repository with

.. code::

    git clone https://github.com/grimme-lab/dxtb
    cd dxtb

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code::

    mamba env create -n torch -f environment.yaml
    mamba activate torch

Install this project with ``pip`` in the environment

.. code::

    pip install .


Dependencies
------------

The following dependencies are required

- `numpy <https://numpy.org/>`__
- `opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/>`__
- `psutil <https://psutil.readthedocs.io/en/latest/>`__
- `scipy <https://www.scipy.org/>`__
- `tad-mctc <https://github.com/tad-mctc/tad-mctc>`__
- `tad-multicharge <https://github.com/tad-mctc/tad-multicharge>`__
- `tad-dftd3 <https://github.com/dftd3/tad-dftd3>`__
- `tad-dftd4 <https://github.com/dftd4/tad-dftd4>`__
- `torch <https://pytorch.org/>`__

For tests, we also require

- `pytest <https://docs.pytest.org/>`__
- `tox <https://docs.pytest.org/>`__
