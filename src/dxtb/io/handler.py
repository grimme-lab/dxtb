"""
I/O: Output Handler
===================

The I/O module contains the singleton `OutputHandler` class that is used to
write output to various streams (console, JSON, ...).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tad_mctc.typing import Any, override

from .output import get_header, get_pytorch_info, get_short_version, get_system_info

__all__ = ["OutputHandler"]


class CustomStreamHandler(logging.StreamHandler):
    """
    A custom stream handler that allows for the addition of a newline at the
    end of the message.
    """

    @override
    def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)

            if getattr(record, "newline", True):
                # issue 35046: merged two stream.writes into one.
                msg += self.terminator

            self.stream.write(msg)
            self.stream.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)


class _OutputHandler:
    """
    Singleton class that handles output to the console and JSON file.
    """

    def __init__(self):
        self.handlers = {}
        self.warnings = []

        self.console_logger = logging.getLogger("console_logger")
        self.setup_console_logger()

        self._verbosity = 5
        self._json_file = "dxtb.json"
        self.json_data = {}

    @property
    def json_file(self) -> str:
        """
        Get the path to the JSON file.

        Returns
        -------
        str
            The path to the JSON file.
        """
        return self._json_file

    @json_file.setter
    def json_file(self, file: str) -> None:
        self._json_file = file

    @property
    def verbosity(self) -> int:
        """
        Get the verbosity level.

        Returns
        -------
        int
            The verbosity level.
        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: int | None) -> None:
        if level is None:
            return

        if not isinstance(level, int):
            raise TypeError("Verbosty level must be an integer.")
        self._verbosity = level

    def temporary_disable_on(self) -> None:
        """Temporarily disable output."""
        self._saved_verbosity = self.verbosity
        self.verbosity = 0

    def temporary_disable_off(self) -> None:
        """Re-enable output."""
        self.verbosity = self._saved_verbosity

    def setup_console_logger(self, level=logging.INFO):
        """
        Setup the console logger.

        Parameters
        ----------
        level : int, optional
            The logging level. Defaults to `logging.INFO`.
        """
        ch = CustomStreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(message)s"))
        self.console_logger.addHandler(ch)
        self.console_logger.setLevel(level)
        self.handlers["console"] = self.console_output

    def setup_json_logger(self):
        """Setup the JSON logger."""
        self.handlers["json"] = self.json_output

        if Path(self.json_file).is_file():
            with open(self.json_file, encoding="utf8") as file:
                self.json_data = json.load(file)

    def console_output(self, data: dict[str, Any]):
        """
        Write data to the console.

        Parameters
        ----------
        data : dict[str, Any]
            The data to write to the console.
        """
        for key, value in data.items():
            formatted_message = self.format_for_console(key, value)
            self.console_logger.info(formatted_message)

    def json_output(self, data: dict[str, Any]):
        """
        Write data to the JSON file.

        Parameters
        ----------
        data : dict[str, Any]
            The data to write to the JSON file.
        """
        self.json_data.update(data)
        with open(self.json_file, "w", encoding="utf8") as file:
            json.dump(self.json_data, file, indent=4)

    def write(self, data: dict[str, Any]):
        """
        Write data to all output streams.

        Parameters
        ----------
        data : dict[str, Any]
            The data to write to the output streams.
        """
        for handler in self.handlers.values():
            handler(data)

    def write_stdout(
        self,
        msg: str,
        verbosity: int = 5,
        newline: bool = True,
    ) -> None:
        """
        Write a message to the console.

        Parameters
        ----------
        msg : str
            The message to write.
        verbosity : int, optional
            The verbosity level at which to write the message. Defaults to 5,
            which is the standard verbosity level between 0 and 10.
        newline : bool, optional
            Whether to add a newline at the end of the message.
            Defaults to `True`.
        """
        if self.verbosity >= verbosity:
            extra = {"newline": newline}
            self.console_logger.info(msg, extra=extra)

    def write_stdout_nf(self, msg: str, verbosity: int = 5) -> None:
        """
        Write a message to the console without a newline.

        Parameters
        ----------
        msg : str
            The message to write.
        verbosity : int, optional
            The verbosity level at which to write the message. The default is 5.
        """
        self.write_stdout(msg, verbosity, newline=False)

    #######################################

    def write_row(self, table_name: str, key: str, row: list[Any]) -> None:
        """
        Write a single row of data to a specified table in both console and JSON output.

        Parameters
        ----------
        table_name : str
            The name of the table to which the row belongs.
        row : List[Any]
            A single row of data to be written.
        iter_number : int
            The iteration number or row identifier.
        """
        self.console_logger.info("  ".join([key] + row))

        if table_name not in self.json_data:
            self.json_data[table_name] = {}
        self.json_data[table_name][key.strip()] = row

        # Update JSON file with the new row
        self.json_output({})

    #######################################

    def warn(self, msg: str) -> None:
        """
        Add a warning message to the list of warnings.

        Parameters
        ----------
        msg : str
            The warning message.
        """
        self.warnings.append(msg)

    def format_for_console(self, title: str, info: dict[str, Any]) -> str:
        """
        Format the data for the console.

        Parameters
        ----------
        title : str
            The title of the data.
        info : dict[str, Any]
            The data to format.

        Returns
        -------
        str
            The formatted data.
        """
        formatted_str = f"{title}\n" + "-" * len(title) + "\n\n"
        for key, value in info.items():
            formatted_str += f"{key.ljust(20)}: {value}\n"
        return formatted_str

    def header(self) -> None:
        """Print the header to the console."""
        if self.verbosity >= 5:
            self.console_logger.info(get_header())
            self.console_logger.info(get_short_version())

    def sysinfo(self) -> None:
        """Print system and PyTorch information to the console."""
        if self.verbosity >= 6:
            self.write(get_system_info())
            self.write(get_pytorch_info())


OutputHandler = _OutputHandler()
