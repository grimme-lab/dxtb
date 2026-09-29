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
  `argparse` reports an invalid choice.
- The old `implicit` code path (`scf/pure/iterator.py`: `scf_pure`,
  `scf_wrapper`, `run_scf`) and all use of xitorch's `equilibrium` and
  `RootFinder` in the SCF were removed. The forward solve still uses xitorch's
  root solvers (default: `broyden1`), so iteration counts are unchanged. The
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
- `implicit` now reports the number of SCF iterations (it returned -1).
- With `scp_mode="charge"`, the extra reconnecting mixing step of the
  implicit SCF uses a damping of 1e-5 (the removed pure implicit path used
  1e-4).
- The Fock matrix is reported in the results of the implicit SCF for
  `scp_mode="fock"`.

### Fixed

- Memory leak of the implicit SCF (reference cycle through PyTorch's C++
  autograd graph via `ctx.fcn` of xitorch's `RootFinder`): the SCF object and
  its graph are freed by reference counting alone.
- `BaseSCF` no longer stores a bound method of itself (`self._fcn`), which
  produced collectable cyclic garbage in every SCF mode (also `full`).

### Known limits

- Derivatives with respect to non-leaf intermediates of autograd leaves miss
  the implicit term; differentiate with respect to leaves.
- Third and higher derivatives through the implicit SCF are not supported.
- Position derivatives of the squared charges and their Hessian-vector
  products on slowly converging systems (24+ iterations) agree with finite
  differences only to ~1e-8 (gradient) and ~1e-6 (Hessian-vector product),
  which is at the noise level of the finite-difference reference.
- Systems with (near-)degenerate frontier orbitals and Fermi smearing
  (e.g., HOMO-LUMO gap of a few mEh) have inexact derivatives of
  non-variational quantities in every SCF mode.
