"""Executable script which runs all unittests."""
import logging
from os import path, scandir
import sys
import unittest


def suite() -> unittest.TestSuite:
    this_directory = path.dirname(path.abspath(__file__))

    # Add the src directory to sys.path so that all imports in the unittests work
    src_directory = path.join(path.abspath(this_directory), "..", "src")
    sys.path.insert(0, src_directory)

    # Suppress logging
    logging.disable(logging.ERROR)

    # Run all tests
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    for entry in scandir(this_directory):
        if not entry.is_dir():
            continue

        if not path.isfile(this_directory + "/" + entry.name + "/__init__.py"):
            continue

        if not entry.name.startswith("test_"):
            continue

        # if entry.name != "test_repulsion":  # and entry.name != "test_ncoord":
        # continue
        print(entry)
        test_suite.addTests(
            loader.discover(
                this_directory + "/" + entry.name, top_level_dir=this_directory
            )
        )

    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
