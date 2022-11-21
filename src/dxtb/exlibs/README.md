# External Libraries

## xitorch

Contains a minimal version of [xitorch](https://github.com/xitorch/xitorch), i.e., all modules that we do not require have been removed.
The library is manually included in order to fix the step size in [xitorch/\_impls/optimize/root/rootsolver.py](xitorch/_impls/optimize/root/rootsolver.py)

```python
s = 0.95
xnew = x + s * dx
```

The source code was altered by all pre-commit hooks.
