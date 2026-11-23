"""Helpers for the notebook builder scripts (python notebooks/_build/build_XX.py)."""

import sys
from pathlib import Path

import nbformat as nbf

SETUP = r'''
import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

FIG = ROOT / "reports/figures"; FIG.mkdir(parents=True, exist_ok=True)
RES = ROOT / "reports/results"; RES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
pd.set_option("display.width", 180); pd.set_option("display.max_columns", 40)
'''


class Builder:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip()))

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip()))

    def write(self, path):
        nb = nbf.v4.new_notebook(cells=self.cells, metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
        nbf.write(nb, path)


def out_path(name):
    return Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / name
