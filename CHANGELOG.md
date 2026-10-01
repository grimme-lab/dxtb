# Changelog

## Unreleased

### Changed: one implicit SCF mode with exact derivatives

- The two implicit SCF modes were replaced by a single mode, `implicit`
  (`scf_mode="implicit"`, alias `"default"`, integer code 1). It applies the
  implicit function theorem with a differentiable adjoint solve, and gives
  first and second derivatives that agree with finite differences for all
  quantities (energy, charges, dipole, ...), including batched and padded
  input.
- **Removed:** the `nonpure` mode (`"nonpure"`, `"non-pure"`, `"old"`,
  `"implicit_old"`, `"implicit_nonpure"`, integer code 2). Passing one of them
  to the Python API raises a `ValueError` that names `implicit` as the
  replacement; the command line (`--scf-mode`) no longer offers `nonpure`, so
  `argparse` reports an invalid choice. The constants
  `labels.SCF_MODE_IMPLICIT_NON_PURE` and `labels.SCF_MODE_IMPLICIT_NON_PURE_STRS`
  are kept as deprecated aliases and lead to the same error.
- The old `implicit` code path (`scf/pure/iterator.py`: `scf_pure`,
  `scf_wrapper`, `run_scf`) and all use of xitorch's `equilibrium` and
  `RootFinder` in the SCF were removed. The forward solve still uses xitorch's
  root solvers (default: `broyden1`); the forward solve is unchanged. The
  vendored xitorch `symeig` remains in use.

### Behaviour changes (these results were wrong before)

- Derivatives of quantities that are not variational (atomic charges, dipole,
  anything that depends on the converged charges) with respect to positions,
  electric fields or parameters were wrong or missing in the old `implicit`
  mode (errors of 1e-4 to 1e-1) and, for second derivatives, approximate in
  `nonpure`. They are now exact. Energy gradients are unchanged (they are
  variational).
- Energy Hessians and other second derivatives that involve the SCF solution
  change accordingly.
- Derivatives with respect to the total charge no longer fail with
  `derivative for aten::heaviside is not implemented`.
- `implicit` now reports the number of SCF iterations (it returned -1), and
  its converged energies agree with the previous non-pure results.
- With `scp_mode="charge"`, the extra reconnecting mixing step of the
  implicit SCF uses a damping of 1e-5 (the removed pure implicit path used
  1e-4).
- The Fock matrix is reported in the results of the implicit SCF for
  `scp_mode="fock"`.
- `bck_options` of the implicit SCF accept `maxiter` (or xitorch's
  `max_niter`), `atol`, `rtol` and `m` (Anderson history). Other keys (e.g.,
  `method`) have no effect and now produce a warning.

### Fixed

- Memory leak of the implicit SCF (reference cycle through PyTorch's C++
  autograd graph via `ctx.fcn` of xitorch's `RootFinder`): the SCF object and
  its graph are freed by reference counting alone.
- `BaseSCF` no longer stores a bound method of itself (`self._fcn`), which
  produced collectable cyclic garbage in every SCF mode (also `full`).
- Batched implicit SCF: a system that receives no gradient (e.g., one row of
  a batched force Jacobian) no longer crashes the backward pass in single
  precision or stalls it in double precision.

### Known limits

- Derivatives with respect to non-leaf intermediates of autograd leaves miss
  the implicit term; differentiate with respect to leaves.
- Third and higher derivatives through the implicit SCF are not supported.
- If the adjoint iterations do not converge, the dense fallback builds one
  matrix for the whole batch (size `(batch * n)^2`, with `n = norb^2` in Fock
  mode), which can exhaust memory for large systems.
- Position derivatives of the squared charges and their Hessian-vector
  products on slowly converging systems (24+ iterations) agree with finite
  differences only to ~1e-8 (gradient) and ~1e-6 (Hessian-vector product),
  which is at the noise level of the finite-difference reference.
- Systems with (near-)degenerate frontier orbitals and Fermi smearing
  (e.g., HOMO-LUMO gap of a few mEh) have inexact derivatives of
  non-variational quantities in every SCF mode.
