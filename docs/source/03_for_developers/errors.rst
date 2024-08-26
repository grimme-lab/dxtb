.. _dev_errors:

Common Errors
=============

The following comprises a list of common errors that might occur during
development and their solutions. If the error occurs frequently, please
raise a custom error in the respective code.


RuntimeError: The linear operator A must be Hermitian
-----------------------------------------------------

.. code-block:: shell

    cond = False, msg = 'The linear operator A must be Hermitian'

        def assert_runtime(cond, msg=""):
            if not cond:
    >           raise RuntimeError(msg)
    E           RuntimeError: The linear operator A must be Hermitian

    src/dxtb/_src/exlibs/xitorch/_utils/assertfuncs.py:49: RuntimeError

This error usually occurs in the later stages of the SCF for atoms. Although
the error message claims that there is a problem with the Hamiltonian matrix,
the error is actually raised because the mixer in `xitorch` produces a
`NaN` value. In fact, the offending code is the following:

.. code-block:: python

    def update(self, x, y):
        dy = y - self.y_prev
        dx = x - self.x_prev # <-- can become exactly zero
        # update Gm
        self._update(x, y, dx, dy, dx.norm(), dy.norm())

        self.y_prev = y
        self.x_prev = x

    def _update(self, x, y, dx, dy, dxnorm, dynorm):
        # keep the rank small
        self._reduce()

        v = self.Gm.rmv(dx)
        c = dx - self.Gm.mv(dy)
        d = v / torch.dot(dy, v) # <-- yields NaN
        self.Gm = self.Gm.append(c, d)

Apparently, this occurs if the convergence criteria are too strict in single
precision. The solution is to increase the convergence criteria to more than
1e-6.


RuntimeError: clone is not supported by NestedIntSymNode
--------------------------------------------------------

This is a bug in PyTorch 2.3.0 and 2.3.1 (see
`PyTorch #128607 <<https://github.com/pytorch/pytorch/issues/128607>`__).
To avoid this error, manually import `torch._dynamo` in the code. For example:

.. code-block:: python

    from tad_mctc._version import __tversion__

    if __tversion__ in ((2, 3, 0), (2, 3, 1)):
        import torch._dynamo
