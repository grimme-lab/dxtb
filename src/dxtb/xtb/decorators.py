"""
xTB: Calculator Decorators
==========================

Decorators for the Calculator class.
"""

from __future__ import annotations

from functools import wraps
from typing import cast

import torch
from tad_mctc.typing import Any, Callable, Tensor, TypeVar

from ..constants import defaults
from ..interaction.external import field as efield
from ..interaction.external import fieldgrad as efield_grad
from ..io import OutputHandler

# class CalculatorFunction(Protocol):
#     def __call__(
#         self: "Calculator",
#         numbers: Tensor,
#         positions: Tensor,
#         chrg: Tensor | float | int = defaults.CHRG,
#         spin: Tensor | float | int | None = defaults.SPIN,
#         **kwargs: Any
#     ) -> tuple[torch.Tensor, Tensor]:
#         ...

F = TypeVar("F", bound=Callable[..., Any])


def requires_positions_grad(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def wrapper(
        self: Calculator,  # type: ignore
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        if not positions.requires_grad:
            raise RuntimeError(
                f"Position tensor needs `requires_grad=True` in '{func.__name__}'."
            )

        return func(self, numbers, positions, chrg, spin, *args, **kwargs)

    return wrapper


def requires_efield(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def wrapper(
        self: Calculator,  # type: ignore
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        if efield.LABEL_EFIELD not in self.interactions.labels:
            raise RuntimeError(
                f"{func.__name__} requires an electric field. Add the "
                f"'{efield.LABEL_EFIELD}' interaction to the Calculator."
            )
        return func(self, numbers, positions, chrg, spin, *args, **kwargs)

    return wrapper


def requires_efield_grad(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def wrapper(
        self: Calculator,  # type: ignore
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        if not ef.field.requires_grad:
            raise RuntimeError(
                f"Field tensor needs `requires_grad=True` in '{func.__name__}'."
            )
        return func(self, numbers, positions, chrg, spin, *args, **kwargs)

    return wrapper


def requires_efg(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def wrapper(
        self: Calculator,  # type: ignore
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        if efield_grad.LABEL_EFIELD_GRAD not in self.interactions.labels:
            raise RuntimeError(
                f"{func.__name__} requires an electric field. Add the "
                f"'{efield_grad.LABEL_EFIELD_GRAD}' interaction to the "
                "Calculator."
            )
        return func(self, numbers, positions, chrg, spin, *args, **kwargs)

    return wrapper


def requires_efg_grad(func: Callable[..., Tensor]) -> Callable[..., Tensor]:
    def wrapper(
        self: Calculator,  # type: ignore
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        efg = self.interactions.get_interaction(efield_grad.LABEL_EFIELD_GRAD)
        if not efg.field_grad.requires_grad:
            raise RuntimeError("Field gradient tensor needs `requires_grad=True`.")
        return func(self, numbers, positions, chrg, spin, **kwargs)

    return wrapper


def _numerical(nograd: bool = False) -> Callable[[F], F]:
    """
    Decorator for numerical differentiation.
    Pass `True` to turns off gradient tracking for the function.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            OutputHandler.temporary_disable_on()
            try:
                if nograd is True:
                    with torch.no_grad():
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            finally:
                OutputHandler.temporary_disable_off()
            return result

        return cast(F, wrapper)

    return decorator


def numerical(func: F) -> F:
    """
    Decorator for numerical differentiation. Turns off gradient tracking.

    .. warning::

        Since this decorator turns off gradient tracking for the function, a
        possible `requires_grad=True` will be lost because the corresponding
        tensor is updated within the numerical differentiation.
        This happens in any electric field related derivatives. If you want to carry out a subsequent calculation with `requires_grad=True`, you have to update the electric field tensor manually with:

        .. code-block:: python

            field_tensor.requires_grad_(True)
            calc.interactions.update_efield(field=field_tensor)
    """
    return _numerical(nograd=True)(func)
