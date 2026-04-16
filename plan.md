# Plan: Fix D4 Float32 Dtype Mismatch

The bug manifests as `RuntimeError: expected scalar type Double but found Float` when running dxtb with `torch.float` (float32), while tad-dftd4's own float32 tests pass. The root cause is dtype contamination at the **dxtb ↔ tad-dftd4 boundary**, not inside tad-dftd4 itself.

## Root Cause

Three related bugs in dxtb's charge/parameter handling allow improperly-typed tensors to reach tad-dftd4:

1. **`EnergyCalculator.singlepoint()`** (`energy.py` line 160) passes the **original** `chrg` argument to `self.classicals.get_energy()` instead of the dtype-converted `_chrg` (created at line 135 via `any_to_tensor(chrg, **self.dd)`). When users pass a plain Python `int` (e.g., `calc.singlepoint(positions, chrg=0)`), the raw `0` reaches `d4.dftd4()` → `get_eeq_charges()` may produce float64 charges → dtype mismatch with float32 positions downstream.

2. **`Dispersion.__init__()`** (`base.py` line 77) stores `self.charge = charge` without any `.to(**self.dd)` conversion, so a wrong-dtype tensor propagates.

3. **`AnalyticalCalculator`** (`analytical.py` line 282) does not forward `charge` to classicals at all, so D4 always falls back to `charge=0.0` — correct dtype but **semantically wrong** for charged systems.

Additionally, while tad-dftd4 passes its own tests, the damping functions in `functions.py` accept params as `Tensor | float | int` and perform arithmetic directly (`a1 * torch.sqrt(radii) + a2`) — a Python float default (inherently float64) could cause silent upcast if the caller doesn't pass tensors.

## Steps

1. **Reproduce the failure**: Run the float32 spinpol test to capture the exact traceback: `pytest test/test_spinpol/test_energy.py -k "torch.float" -x --tb=long`. Also run `pytest test/test_singlepoint/test_energy.py -k "gfn2 and torch.float" -x --tb=long` and `pytest test/test_classical/test_dispersion/test_d4sc.py -k "torch.float" -x --tb=long`.

2. **Fix charge dtype in `EnergyCalculator.singlepoint()`** (`energy.py` line 160): Change `charge=chrg` to `charge=_chrg` so the properly-typed tensor is forwarded to classical contributions.

3. **Add defensive dtype cast in `Dispersion.__init__()`** (`base.py` line 77): Change `self.charge = charge` to `self.charge = charge.to(**self.dd) if isinstance(charge, Tensor) else charge`.

4. **Add defensive dtype cast in `DispersionD4.get_energy()`** (`d4.py` line 197): After `charge = kwargs.pop("charge", self.charge)`, add `charge = charge.to(**self.dd) if isinstance(charge, Tensor) else torch.tensor(charge if charge is not None else 0.0, **self.dd)`.

5. **Forward charge in analytical forces path** (`analytical.py` line 282): Pass `charge=chrg` to `self.classicals.get_energy(positions, ccaches, charge=chrg)` so non-zero charged systems compute D4 correctly under gradient computation.

6. **Harden tad-dftd4 damping function** (`functions.py` `RationalDamping._f`): In `_f()`, convert `a1` and `a2` to match `distances.dtype` before arithmetic: `a1 = torch.as_tensor(a1, dtype=distances.dtype, device=distances.device)` (and similarly for `a2`). This prevents silent float64 upcast if a Python float or wrong-dtype tensor leaks through.

7. **Harden tad-dftd4 `dispersion2()` params** (`twobody.py` lines 167-168): After `s6 = param.get("s6", ...)` and `s8 = param.get("s8", ...)`, add `.to(**dd)` to enforce dtype from positions: `s6 = param.get("s6", torch.tensor(defaults.S6, **dd)).to(**dd)` etc.

8. **Verify**: Re-run the full test suites:
   - `pytest test/test_spinpol/ -x` (all dtypes)
   - `pytest test/test_singlepoint/test_energy.py -x`
   - `pytest test/test_classical/test_dispersion/ -x`
   - In tad-dftd4: `pytest test/ -x` to verify no regressions

## Verification

- All dxtb float32 tests pass (spinpol, singlepoint, D4, D4SC)
- All tad-dftd4 tests still pass (float32 and float64)
- Manual check: `python -c "import dxtb; calc = dxtb.Calculator(..., dtype=torch.float); calc.singlepoint(pos, chrg=0)"` runs without dtype errors

## Decisions

- Fix in **both** dxtb and tad-dftd4: dxtb should send clean types; tad-dftd4 should defensively cast params
- Steps 6-7 (tad-dftd4 hardening) are secondary — the primary bugs are in dxtb steps 2-5
- Use `.to(**dd)` pattern rather than `.type()` to handle both device and dtype in one call
