"""Copy outputs from an executed notebook into a freshly generated one.

Code cells are matched by identical source, so only markdown (explanations, insights) may differ.
Lets prose be updated after an expensive run without re-executing:

    python notebooks/_build/merge_outputs.py executed.ipynb generated.ipynb [out.ipynb]
"""

import sys

import nbformat

src, gen = nbformat.read(sys.argv[1], 4), nbformat.read(sys.argv[2], 4)
executed = {}
for c in src.cells:
    if c.cell_type == "code":
        executed.setdefault(c.source, c)
missing = 0
for c in gen.cells:
    if c.cell_type == "code":
        if c.source in executed:
            c.outputs, c.execution_count = executed[c.source].outputs, executed[c.source].execution_count
        else:
            missing += 1
gen.metadata = src.metadata
nbformat.write(gen, sys.argv[3] if len(sys.argv) > 3 else sys.argv[2])
print(f"merged; {missing} code cells without outputs (changed source)")
