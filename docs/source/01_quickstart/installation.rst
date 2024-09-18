.. _quickstart-installation:

Installation
============

pip
---

.. image:: https://img.shields.io/pypi/v/dxtb
    :target: https://pypi.org/project/dxtb/
    :alt: PyPI Version

*dxtb* can easily be installed with ``pip``.

.. code-block:: shell

    pip install dxtb

Installing the libcint interface is highly recommended, as it is significantly
faster than the pure PyTorch implementation and provides access to higher-order
multipole integrals.


conda
-----

.. image:: https://img.shields.io/conda/vn/conda-forge/dxtb.svg
   :target: https://anaconda.org/conda-forge/dxtb
   :alt: Conda Version

*dxtb* is also available on `conda <https://conda.io/>`__.

.. code-block:: shell

    mamba install dxtb


Don't forget to install the libcint interface (not on conda) via ``pip install tad-libcint``.


From source
-----------

This project is hosted on GitHub at `grimme-lab/dxtb <https://github.com/grimme-lab/dxtb>`__.
Obtain the source by cloning the repository with

.. code-block:: shell

    git clone https://github.com/grimme-lab/dxtb
    cd dxtb

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code-block:: shell

    mamba env create -n torch -f environment.yaml
    mamba activate torch

Install this project with ``pip`` in the environment

.. code-block:: shell

    pip install .


Without pip
-----------

If you want to install the package without pip, start by cloning the repository.

.. code-block:: shell

    DEST=/opt/software
    git clone https://github.com/grimme-lab/dxtb $DEST/dxtb

Next, add ``<path to dxtb>/dxtb/src`` to your ``$PYTHONPATH`` environment variable.
For the command line interface, add ``<path to dxtb>/dxtb/bin`` to your ``$PATH`` environment variable.

.. code-block:: shell

    export PYTHONPATH=$PYTHONPATH:$DEST/dxtb/src
    export PATH=$PATH:$DEST/dxtb/bin


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
- `tad-libcint <https://github.com/tad-mctc/tad-libcint>`__
- `torch <https://pytorch.org/>`__

For tests, we also require

- `pytest <https://docs.pytest.org/>`__
- `pyscf <https://pyscf.org/>`__
- `tox <https://docs.pytest.org/>`__
