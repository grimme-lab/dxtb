"""
External Fields
===============

This module contains implementations of external fields that interact with the
charge density. The fields object are interactions that can straightfowradly be
supplied to the SCF.
"""
from .field import ElectricField, new_efield
