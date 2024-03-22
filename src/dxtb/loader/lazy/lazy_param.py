from __future__ import annotations

from ..._types import Any, PathLike


class LazyLoaderParam:
    """
    A lazy loader class for loading TOML parametrization files as needed.

    This class is designed to delay the loading of a TOML file until an
    attribute from the file is accessed. It dynamically loads and parses the
    TOML file, initializing a `Param` object with the parsed data only upon
    attribute access.

    Parameters
    ----------
    filepath : PathLike
        The file path to the TOML file that needs to be lazily loaded.

    Attributes
    ----------
    filepath : PathLike
        Stores the file path of the TOML file.
    _loaded : Param or None
        Stores the loaded `Param` object after the first attribute access.

    Methods
    -------
    __getattr__(item: Any) -> Any
        Overridden method to load the TOML file and access attributes of the
        `Param` object.
    """

    def __init__(self, filepath: PathLike) -> None:
        """
        Initializes the LazyLoaderParam with the specified file path.

        Parameters
        ----------
        filepath : PathLike
            The file path to the TOML file that needs to be lazily loaded.
        """
        self.filepath = filepath
        self._loaded = None

    def __getattr__(self, item: Any) -> Any:
        """
        Loads the TOML file and initializes the `Param` object upon first
        attribute access.
        Subsequent accesses will use the already loaded `Param` object.

        Parameters
        ----------
        item : Any
            The attribute name to be accessed from the `Param` object.

        Returns
        -------
        Any
            The value of the attribute from the `Param` object.
        """
        if self._loaded is None:
            import tomli as toml

            from dxtb.param.base import Param

            with open(self.filepath, "rb") as fd:
                self._loaded = Param(**toml.load(fd))

            del toml

        return getattr(self._loaded, item)