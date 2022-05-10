"""
Run tests for repulsion contribution.

(Note that the analytical gradient tests fail for `torch.float32`.)
"""

from typing import Callable, Literal, Union
from xtbml.typing import Tensor

import pytest
import torch

from xtbml.exlibs.tbmalt import batch
from xtbml.repulsion import RepulsionFactory
from xtbml.utils import symbol2number
from xtbml.param.gfn1 import GFN1_XTB

from samples import mb16_43
from samples import amino20x4


class Setup:
    """Setup class to define constants for test class."""

    cutoff: Tensor = torch.tensor(25.0)
    """Cutoff for repulsion calculation."""


class TestRepulsion(Setup):
    """Testing the calculation of repulsion energy and gradients."""

    @classmethod
    def setup_class(cls):
        print(cls.__name__)

    def repulsion_gfn1(
        self, numbers: Tensor, positions: Tensor, req_grad: bool = False
    ) -> RepulsionFactory:
        """Factory for repulsion construction based on GFN1-xTB"""

        # setup repulsion
        repulsion = RepulsionFactory(
            numbers=numbers, positions=positions, req_grad=req_grad, cutoff=self.cutoff
        )
        repulsion.setup(GFN1_XTB.element, GFN1_XTB.repulsion.effective)

        return repulsion

    def repulsion_gfn2(
        self, numbers: Tensor, positions: Tensor, req_grad: bool
    ) -> RepulsionFactory:
        """Factory for repulsion construction based on GFN2-xTB"""

        # setup repulsion
        repulsion = RepulsionFactory(
            numbers=numbers, positions=positions, req_grad=req_grad, cutoff=self.cutoff
        )
        # repulsion.setup(GFN2_XTB.element, GFN2_XTB.repulsion.effective)

        return repulsion

    def base_test(
        self,
        numbers: Tensor,
        positions: Tensor,
        repulsion_factory: Callable,
        reference_energy: Union[Tensor, None],
        reference_gradient: Union[Tensor, None],
        dtype: torch.dtype,
        repulsion_mode: Literal["full", "pair", "atom", "scalar"] = "pair",
    ) -> None:
        """Wrapper for testing versus reference energy and gradient"""

        atol = 1.0e-6 if dtype == torch.float32 else 1.0e-8
        rtol = 1.0e-4 if dtype == torch.float32 else 1.0e-5

        # factory to produce repulsion objects
        repulsion: RepulsionFactory = repulsion_factory(numbers, positions)
        e, gradient = repulsion.get_engrad(calc_gradient=True, mode=repulsion_mode)

        # sum up for comparison with reference
        if repulsion_mode == "pair":
            energy = torch.sum(e, dim=(-2, -1))
        elif repulsion_mode == "atom":
            energy = torch.sum(e, dim=-1)
        elif repulsion_mode == "full":
            energy = 0.5 * torch.sum(e, dim=(-2, -1))
        elif repulsion_mode == "scalar":
            energy = e
        else:
            raise RuntimeError(f"Unknown mode '{repulsion_mode}' for repulsion tensor.")

        # test against reference values
        if reference_energy is not None:
            assert torch.allclose(
                energy,
                reference_energy,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
        if reference_gradient is not None:
            assert torch.allclose(
                gradient,
                reference_gradient,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )

    def calc_numerical_gradient(
        self,
        repulsion: RepulsionFactory,
        dtype: torch.dtype,
    ) -> Tensor:
        """Calculate gradient numerically for reference."""

        n_atoms = repulsion.positions.shape[0]

        # numerical gradient
        gradient = torch.zeros(n_atoms, 3, dtype=dtype)
        step = 1.0e-6

        for i in range(n_atoms):
            for j in range(3):
                er, el = 0.0, 0.0
                repulsion.positions[i, j] += step
                er, _ = repulsion.get_engrad(calc_gradient=False)
                er = torch.sum(er, dim=(-2, -1))
                repulsion.positions[i, j] -= 2 * step
                el, _ = repulsion.get_engrad(calc_gradient=False)
                el = torch.sum(el, dim=(-2, -1))
                repulsion.positions[i, j] += step
                gradient[i, j] = 0.5 * (er - el) / step

        return gradient

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_modes(self, dtype: torch.dtype):
        sample = mb16_43["01"]

        atomic_numbers = symbol2number(sample["symbols"])
        positions = sample["positions"].type(dtype)

        reference_energy = torch.tensor(sample["ref"]["gfn1"]["repulsion"], dtype=dtype)
        reference_gradient = None

        self.base_test(
            numbers=atomic_numbers,
            positions=positions,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
            dtype=dtype,
            repulsion_mode="pair",
        )
        self.base_test(
            numbers=atomic_numbers,
            positions=positions,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
            dtype=dtype,
            repulsion_mode="atom",
        )
        self.base_test(
            numbers=atomic_numbers,
            positions=positions,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
            dtype=dtype,
            repulsion_mode="full",
        )
        self.base_test(
            numbers=atomic_numbers,
            positions=positions,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
            dtype=dtype,
            repulsion_mode="scalar",
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_gfn1_mb1643_01(self, dtype: torch.dtype):
        sample = mb16_43["01"]

        atomic_numbers = symbol2number(sample["symbols"])
        positions = sample["positions"].type(dtype)

        reference_energy = torch.tensor(sample["ref"]["gfn1"]["repulsion"], dtype=dtype)
        reference_gradient = None

        self.base_test(
            numbers=atomic_numbers,
            positions=positions,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
            dtype=dtype,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_gfn1_mb1643_02(self, dtype: torch.dtype):
        sample = mb16_43["02"]

        atomic_numbers = symbol2number(sample["symbols"])
        positions = sample["positions"].type(dtype)

        reference_energy = torch.tensor(sample["ref"]["gfn1"]["repulsion"], dtype=dtype)
        reference_gradient = None

        self.base_test(
            numbers=atomic_numbers,
            positions=positions,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
            dtype=dtype,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_gfn1_Amino20x4_LYS_xao(self, dtype: torch.dtype):
        sample = amino20x4["LYS_xao"]

        atomic_numbers = symbol2number(sample["symbols"])
        positions = sample["positions"].type(dtype)

        reference_energy = torch.tensor(sample["ref"]["gfn1"]["repulsion"], dtype=dtype)
        reference_gradient = None

        self.base_test(
            numbers=atomic_numbers,
            positions=positions,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
            dtype=dtype,
        )

    @pytest.mark.parametrize("dtype", [torch.float64])
    def test_gfn1_grad_mb1643_03(self, dtype: torch.dtype):
        sample = mb16_43["03"]

        atomic_numbers = symbol2number(sample["symbols"])
        positions = sample["positions"].type(dtype)

        # get reference gradient
        repulsion = self.repulsion_gfn1(atomic_numbers, positions)
        numerical_gradient = self.calc_numerical_gradient(repulsion, dtype)

        self.base_test(
            numbers=atomic_numbers,
            positions=positions,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=None,
            reference_gradient=numerical_gradient,
            dtype=dtype,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_gfn1_batch(self, dtype: torch.dtype):
        sample1, sample2 = mb16_43["01"], mb16_43["SiH4"]
        numbers: Tensor = batch.pack(
            (
                symbol2number(sample1["symbols"]),
                symbol2number(sample2["symbols"]),
            )
        )
        positions: Tensor = batch.pack(
            (
                sample1["positions"].type(dtype),
                sample2["positions"].type(dtype),
            )
        )

        reference_energy = torch.tensor(
            [
                sample1["ref"]["gfn1"]["repulsion"],
                sample2["ref"]["gfn1"]["repulsion"],
            ],
            dtype=dtype,
        )
        reference_gradient = None

        self.base_test(
            numbers=numbers,
            positions=positions,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
            dtype=dtype,
        )

    @pytest.mark.parametrize("dtype", [torch.float64])
    def test_gfn1_batch_grad(self, dtype: torch.dtype):
        sample1, sample2 = mb16_43["01"], mb16_43["SiH4"]
        numbers: Tensor = batch.pack(
            (
                symbol2number(sample1["symbols"]),
                symbol2number(sample2["symbols"]),
            )
        )
        positions: Tensor = batch.pack(
            (
                sample1["positions"].type(dtype),
                sample2["positions"].type(dtype),
            )
        )

        # get reference gradients by just looping over geometries
        grads = []
        for number, position in zip(numbers, positions):
            repulsion = self.repulsion_gfn1(number, position)
            grad = self.calc_numerical_gradient(repulsion, dtype)
            grads.append(grad)

        numerical_gradient = batch.pack(grads)

        self.base_test(
            numbers=numbers,
            positions=positions,
            repulsion_factory=self.repulsion_gfn1,
            reference_energy=None,
            reference_gradient=numerical_gradient,
            dtype=dtype,
        )

    @pytest.mark.grad
    @pytest.mark.parametrize("dtype", [torch.float64])
    def test_param_grad(self, dtype: torch.dtype):
        """
        Check a single analytical gradient of `RepulsionFactory.alpha` against numerical gradient from `torch.autograd.gradcheck`.

        Args:
            dtype (torch.dtype): Numerical precision.

        Note:
            Although `torch.float32` raises a warning that the gradient check without double precision will fail, it actually works here.
        """
        sample = mb16_43["01"]

        numbers = symbol2number(sample["symbols"])
        positions = sample["positions"].type(dtype)

        def func(*_):
            repulsion = self.repulsion_gfn1(numbers, positions, True)
            energy, _ = repulsion.get_engrad()
            return energy

        repulsion = self.repulsion_gfn1(numbers, positions, True)
        param = (repulsion.alpha, repulsion.zeff, repulsion.kexp)
        assert torch.autograd.gradcheck(func, param)

    @pytest.mark.grad
    @pytest.mark.parametrize("dtype", [torch.float64])
    def test_param_grad_batch(self, dtype: torch.dtype):
        """
        Check batch analytical gradient against numerical gradient from `torch.autograd.gradcheck`.

        Args:
            dtype (torch.dtype): Numerical precision.

        Note:
            Although `torch.float32` raises a warning that the gradient check without double precision will fail, it actually works here.
        """

        sample1, sample2 = mb16_43["01"], mb16_43["SiH4"]
        numbers: Tensor = batch.pack(
            (
                symbol2number(sample1["symbols"]),
                symbol2number(sample2["symbols"]),
            )
        )
        positions: Tensor = batch.pack(
            (
                sample1["positions"].type(dtype),
                sample2["positions"].type(dtype),
            )
        )

        def func(*_):
            repulsion = self.repulsion_gfn1(numbers, positions, True)
            energy, _ = repulsion.get_engrad()
            return energy

        repulsion = self.repulsion_gfn1(numbers, positions, True)
        param = (repulsion.alpha, repulsion.zeff, repulsion.kexp)
        assert torch.autograd.gradcheck(func, param)

    @pytest.mark.grad
    @pytest.mark.parametrize("dtype", [torch.float64])
    def test_positions_grad(self, dtype):
        """
        Check a single analytical gradient of `RepulsionFactory.positions` against numerical gradient from `torch.autograd.gradcheck`.

        Args:
            dtype (torch.dtype): Numerical precision.

        Note:
            Fails for `torch.float32`.
        """
        sample = mb16_43["01"]

        numbers = symbol2number(sample["symbols"])
        positions = sample["positions"].type(dtype)
        positions.requires_grad_(True)

        def func(positions):
            repulsion = self.repulsion_gfn1(numbers, positions, True)
            energy, _ = repulsion.get_engrad()
            return energy

        assert torch.autograd.gradcheck(func, positions)

    @pytest.mark.grad
    @pytest.mark.parametrize("dtype", [torch.float64])
    def test_positions_grad_batch(self, dtype: torch.dtype):
        """
        Check batch analytical gradient against numerical gradient from `torch.autograd.gradcheck`.

        Args:
            dtype (torch.dtype): Numerical precision.

        Note:
            Fails for `torch.float32`.
        """

        sample1, sample2 = mb16_43["01"], mb16_43["SiH4"]
        numbers: Tensor = batch.pack(
            (
                symbol2number(sample1["symbols"]),
                symbol2number(sample2["symbols"]),
            )
        )
        positions: Tensor = batch.pack(
            (
                sample1["positions"].type(dtype).requires_grad_(True),
                sample2["positions"].type(dtype).requires_grad_(True),
            )
        )

        def func(positions):
            repulsion = self.repulsion_gfn1(numbers, positions, True)
            energy, _ = repulsion.get_engrad()
            return energy

        assert torch.autograd.gradcheck(func, positions)

    # REQUIRES GFN2-XTB PARAMETRIZATION
    """
    def test_mb1643_02_gfn2(self):
        sample = data["MB16_43_02"]

        atomic_numbers = symbol2number(sample["symbols"])
        geometry = Geometry(atomic_numbers=atomic_numbers, positions=sample["positions"])

        reference_energy = torch.tensor(0.10745931926703985)
        reference_gradient = None

        self.base_test(
            geometry=geometry,
            repulsion_factory=self.repulsion_gfn2,
            reference_energy=reference_energy,
            reference_gradient=reference_gradient,
        )

    def test_mb1643_04_gfn2(self):
        sample = data["MB16_43_04"]

        atomic_numbers = symbol2number(sample["symbols"])
        geometry = Geometry(atomic_numbers=atomic_numbers, positions=sample["positions"])

        # get reference gradient
        repulsion = self.repulsion_gfn2(geometry)
        numerical_gradient = self.calc_numerical_gradient(
            geometry, repulsion, self.cutoff
        )

        self.base_test(
            geometry=geometry,
            repulsion_factory=self.repulsion_gfn2,
            reference_energy=None,
            reference_gradient=numerical_gradient,
        )
    """

    # REQUIRES PBC implementation
    """def test_uracil_gfn2(self):
        sample = data["uracil"]

        atomic_numbers = symbol2number(sample["symbols"])
        geometry = Geometry(atomic_numbers=atomic_numbers,positions=sample["positions"])

        reference_energy = torch.tensor(1.0401472262740301)
        reference_gradient = None
        self.base_test(geometry=geometry, 
                    repulsion_factory=self.repulsion_gfn2,
                    reference_energy=reference_energy, 
                    reference_gradient=reference_gradient)"""

    # REQUIRES PBC implementation
    """subroutine test_g_effective_urea(error)

    !> Error handling
    type(error_type), allocatable, intent(out) :: error

    type(structure_type) :: mol

    call get_structure(mol, "X23", "urea")
    call test_numgrad(error, mol, make_repulsion2)

    end subroutine test_g_effective_urea"""


# REQUIRES PBC implementation (strain and sigma not implemented yet)
"""
subroutine test_s_effective_m05(error)

   !> Error handling
   type(error_type), allocatable, intent(out) :: error

   type(structure_type) :: mol

   call get_structure(mol, "MB16-43", "05")
   call test_numsigma(error, mol, make_repulsion1)

end subroutine test_s_effective_m05


subroutine test_s_effective_m06(error)

   !> Error handling
   type(error_type), allocatable, intent(out) :: error

   type(structure_type) :: mol

   call get_structure(mol, "MB16-43", "06")
   call test_numsigma(error, mol, make_repulsion2)

end subroutine test_s_effective_m06


subroutine test_s_effective_succinic(error)

   !> Error handling
   type(error_type), allocatable, intent(out) :: error

   type(structure_type) :: mol

   call get_structure(mol, "X23", "succinic")
   call test_numsigma(error, mol, make_repulsion2)

end subroutine test_s_effective_succinic

end module test_repulsion"""
