"""
Components: Base Class
======================

Base class for all tight-binding components.
"""

from __future__ import annotations

import torch
from tad_mctc.typing import Any, Tensor, TensorLike


class Component(TensorLike):
    """
    Base class for all tight-binding terms.
    """

    label: str
    """Label for the tight-binding component."""

    __slots__ = ["label"]

    class Cache(TensorLike):
        """Cache of a component."""

        def __init__(
            self,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
            super().__init__(device, dtype)

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.label = self.__class__.__name__

    def update(self, **kwargs: Any) -> None:
        """
        Update the attributes of the `Component` instance.

        This method updates the attributes of the `Component` instance based
        on the provided keyword arguments. Only the attributes defined in
        `__slots__` can be updated.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Keyword arguments where keys are attribute names and values are the
            new values for those attributes.
            Valid keys are those defined in `__slots__` of this class.

        Raises
        ------
        AttributeError
            If any key in kwargs is not an attribute defined in `__slots__`.

        Examples
        --------
        >>> import torch
        >>> import dxtb.interaction.external.field import ElectricField
        >>> ef = ElectricField(field=torch.tensor([0.0, 0.0, 0.0]))
        >>> ef.update(field=torch.tensor([1.0, 0.0, 0.0]))
        """
        for key, value in kwargs.items():
            if key is None:
                continue

            if key in self.__slots__:
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"Cannot update '{key}' of the '{self.__class__.__name__}' "
                    "interaction. Invalid attribute."
                )

    def reset(self) -> None:
        """
        Reset the tensor attributes of the `Component` instance to their
        original states or to specified values.

        This method iterates through the attributes defined in `__slots__` and
        resets any tensor attributes to a detached clone of their original
        state. The `requires_grad` status of each tensor is preserved.

        Examples
        --------
        >>> import torch
        >>> import dxtb.interaction.external.field import ElectricField
        >>> ef = ElectricField(field=torch.tensor([0.0, 0.0, 0.0]))
        >>> ef.reset()

        Notes
        -----
        Only tensor attributes defined in `__slots__` are reset. Non-tensor
        attributes are ignored. Attempting to reset an attribute not defined in
        `__slots__` or providing a non-tensor value in `kwargs` will not raise
        an error; the method will simply ignore these cases and proceed with
        the reset operation for valid tensor attributes.
        """
        for slot in self.__slots__:
            attr = getattr(self, slot)
            if isinstance(attr, Tensor):
                reset = attr.detach().clone()
                reset.requires_grad = attr.requires_grad

                setattr(self, slot, reset)
