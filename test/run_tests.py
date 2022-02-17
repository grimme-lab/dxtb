"""Executable script which runs all unittests."""
import logging
import os
import os.path as op
import sys
import unittest

if __name__ == '__main__':
    # Add the src directory to sys.path so that all imports in the unittests work
    this_directory = op.dirname(op.abspath(__file__))
    src_directory = op.join(op.abspath(this_directory), "..", "src")
    sys.path.insert(0, src_directory)

    # Suppress logging
    logging.disable(logging.ERROR)

    # Run all tests
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    for entry in os.scandir(this_directory):
        if (entry.is_file() and str(entry.name).startswith("test_")):
            print(entry)
            test_suite.addTests(loader.discover(this_directory, top_level_dir=this_directory))
    runner = unittest.TextTestRunner()
    runner.run(test_suite)
