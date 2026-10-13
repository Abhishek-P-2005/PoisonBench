"""Lets `pytest` run from anywhere: puts the repo root on sys.path."""
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.getLogger("defenses").setLevel(logging.ERROR)
