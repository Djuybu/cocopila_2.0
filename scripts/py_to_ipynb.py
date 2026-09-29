"""Utility to convert # %% percent format Python scripts to Jupyter Notebook (.ipynb)."""
import json
from pathlib import Path
import sys


def convert_py_to_ipynb(py_path, ipynb_path=None):
    py_path = Path(py_path)
    if ipynb_path is None:
        ipynb_path = py_path.with_suffix(".ipynb")
    else:
        ipynb_path = Path(ipynb_path)

    text = py_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    cells = []
    current_type = None
    current_lines = []

    def flush_cell():
        nonlocal current_type, current_lines
        if current_type and current_lines:
            # Clean leading/trailing blank lines
            cell = {
                "cell_type": current_type,
                "metadata": {},
                "source": current_lines,
            }
            if current_type == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            cells.append(cell)
        current_type = None
        current_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# %% [markdown]"):
            flush_cell()
            current_type = "markdown"
        elif stripped.startswith("# %%"):
            flush_cell()
            current_type = "code"
            # Optional cell name in header
            name = stripped[4:].strip()
            if name and not name.startswith("["):
                current_lines.append(f"# {name}\n")
        else:
            if current_type is None:
                # Default to code if no marker yet
                current_type = "code"
            if current_type == "markdown":
                # Strip leading '# ' or '#' from markdown lines
                if line.startswith("# "):
                    current_lines.append(line[2:])
                elif line.startswith("#\n"):
                    current_lines.append("\n")
                elif line.startswith("#"):
                    current_lines.append(line[1:])
                else:
                    current_lines.append(line)
            else:
                current_lines.append(line)

    flush_cell()

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    ipynb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Generated notebook: {ipynb_path} with {len(cells)} cells.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert # %% format Python script to Jupyter Notebook (.ipynb)")
    parser.add_argument("py_file", type=Path, help="Path to source .py file")
    parser.add_argument("ipynb_file", type=Path, nargs="?", default=None, help="Optional path to output .ipynb file")
    args = parser.parse_args()
    convert_py_to_ipynb(args.py_file, args.ipynb_file)
