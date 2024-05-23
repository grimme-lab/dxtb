# External Libraries

## xitorch

Contains a minimal version of [xitorch](https://github.com/xitorch/xitorch), i.e., all modules that we do not require have been removed.
The library is manually included in order to fix the step size in [xitorch/\_impls/optimize/root/rootsolver.py](xitorch/_impls/optimize/root/rootsolver.py)

```python
s = 0.95
xnew = x + s * dx
```

The source code was altered by all pre-commit hooks.

## scipy

Just a lazily imported version of *scipy*.

## libcint

An interface to *libcint* with automatic differentiation capabilities.
The library is available as a standalone at [tad-mctc/tad-libcint](https://github.com/tad-mctc/tad-libcint)

Acknowledgements: This library is heavily inspired by [diffqc/dqc](https://github.com/diffqc/dqc).
