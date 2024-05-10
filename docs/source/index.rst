.. dxtb documentation master file, created by
   sphinx-quickstart on Mon Apr 29 15:30:04 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :hidden:
   :caption: Quickstart
   :maxdepth: 2

   Installation <01_quickstart/installation>
   Getting Started <01_quickstart/getting_started>
   CLI <01_quickstart/cli>

.. toctree::
   :hidden:
   :caption: In Depth
   :maxdepth: 2

   Calculators <02_indepth/calculators>
   Components <02_indepth/components>
   Integrals <02_indepth/integrals>

.. toctree::
   :hidden:
   :caption: For Developers
   :maxdepth: 2

   Installation <03_for_developers/installation>
   Testing <03_for_developers/testing>
   Style <03_for_developers/style>

.. toctree::
    :hidden:
    :caption: About
    :maxdepth: 2

    Literature <about/literature>
    Related Works <about/related>
    License <about/license>

.. toctree::
    :hidden:
    :caption: Module Reference
    :maxdepth: 0

    modules

dxtb - Fully Differentiable Extended Tight-Binding
==================================================

This project provides a PyTorch-based fully differentiable implementation of the semi-empirical extended tight-binding (xTB) methods.

Introduction
------------

The xTB methods (GFNn-xTB) are a series of semi-empirical quantum chemical methods that provide a good balance between accuracy and computational cost.
For more details and the original Fortran implementation, check out the `GitHub repository <https://github.com/grimme-lab/xtb>`__ and the `documentation <https://xtb-docs.readthedocs.io/>`__.

With *dxtb*, we provide a re-implementation of the xTB methods in PyTorch, which allows for automatic differentiation and seamless integration into machine learning frameworks.

If you use *dxtb* in your research, please cite the following paper:

- dxtb: M. Friede, C. Hölzer, S. Ehlert, S. Grimme, *dxtb -- An Efficient and Fully Differentiable Framework for Extended Tight-Binding*, *J. Chem. Phys.*, **2024**

.. admonition:: BibTeX
   :class: toggle

   .. code-block:: bibtex

       @article{dxtb,
         title = {dxtb -- An Efficient and Fully Differentiable Framework for Extended Tight-Binding},
         author = {Friede, Marvin and Hölzer, Christian and Ehlert, Sebastian and Grimme, Stefan},
         journal = {Journal of Chemical Physics},
         volume = {},
         number = {},
         pages = {},
         year = {2024},
       }


Quicklinks
----------

- :ref:`quickstart-installation`
- :ref:`quickstart-getting-started`
- :ref:`about-literature`
