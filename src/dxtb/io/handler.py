import json
import logging
from pathlib import Path

from .output import get_header, get_pytorch_info, get_short_version, get_system_info

__all__ = ["OutputHandler"]


class _OutputHandler:
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
        return self._json_file

    @json_file.setter
    def json_file(self, file: str) -> None:
        self._json_file = file

    @property
    def verbosity(self) -> int:
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: int) -> None:
        if not isinstance(level, int):
            raise TypeError("Verbosty level must be an integer.")
        self._verbosity = level

    def setup_console_logger(self, level=logging.INFO):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(message)s"))
        self.console_logger.addHandler(ch)
        self.console_logger.setLevel(level)
        self.handlers["console"] = self.console_output

    def setup_json_logger(self):
        self.handlers["json"] = self.json_output

        if Path(self.json_file).is_file():
            with open(self.json_file) as file:
                self.json_data = json.load(file)

    def console_output(self, data):
        for key, value in data.items():
            formatted_message = self.format_for_console(key, value)
            self.console_logger.info(formatted_message)

    def json_output(self, data):
        self.json_data.update(data)
        with open(self.json_file, "w") as file:
            json.dump(self.json_data, file, indent=4)

    def write(self, data):
        for handler in self.handlers.values():
            handler(data)

    def write_stdout(self, msg: str, verbosity: int = 5) -> None:
        if self.verbosity >= verbosity:
            self.console_logger.info(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def format_for_console(self, title, info):
        formatted_str = f"{title}\n" + "-" * len(title) + "\n\n"
        for key, value in info.items():
            formatted_str += f"{key.ljust(17)}: {value}\n"
        return formatted_str

    def header(self) -> None:
        if self.verbosity >= 5:
            self.console_logger.info(get_header())
            self.console_logger.info(get_short_version())

    def sysinfo(self) -> None:
        if self.verbosity >= 6:
            self.write(get_system_info())
            self.write(get_pytorch_info())


OutputHandler = _OutputHandler()
