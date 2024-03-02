from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dxtb.exlibs import scipy as scipy
    from dxtb.exlibs import xitorch as xitorch
else:
    import dxtb.loader.lazy as _lazy

    __getattr__, __dir__, __all__ = _lazy.attach_module(
        __name__,
        ["scipy", "xitorch"],
    )

    del _lazy

del TYPE_CHECKING
