.. dxtb documentation master file, created by
   sphinx-quickstart on Mon Apr 29 15:30:04 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:
   :caption: Quickstart
   :maxdepth: 2

   Installation <01_quickstart/installation>
   Examples <01_quickstart/examples>

.. toctree::
   :hidden:
   :caption: For Developers
   :maxdepth: 2

   Installation <02_for_developers/installation>
   Testing <02_for_developers/testing>
   Style <02_for_developers/style>
   New Components <02_for_developers/extending>



dxtb - Fully Differentiable Extended Tight-Binding
==================================================

This project provides a fully differentiable implementation of the extended tight binding (xTB) Hamiltonian.

To obtain *dxtb*, check out the :ref:`quickstart_installation` instructions.

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
