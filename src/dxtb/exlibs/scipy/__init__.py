from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dxtb.exlibs.scipy import constants as constants
    from dxtb.exlibs.scipy import sparse as sparse
else:
    import dxtb.loader.lazy as _lazy

    __getattr__, __dir__, __all__ = _lazy.attach_module(
        __name__,
        ["constants", "sparse"],
    )

    del _lazy

del TYPE_CHECKING
