"""
Timer
=====

Isolated timer class. Upon importing this subpackage the total timer will start
and thanks to no dependencies on PyTorch, it can reliably measure the total
execution time if imported before PyTotrch.
"""
from .timer import *
