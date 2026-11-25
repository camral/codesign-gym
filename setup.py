"""Compatibility shim.

All packaging metadata lives in ``pyproject.toml``; this file exists only so
that legacy tooling (``python setup.py``, older pip/editable workflows) keeps
working.
"""

from setuptools import setup

setup()
