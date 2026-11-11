"""Builds the .ipynb from build/parts/*.py (cells separated by '# %%' / '# %% [markdown]') and executes it."""
import sys, re, glob, time
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB_NAME = "Code.ipynb"

def parse_parts():
    cells = []
    for f in sorted(glob.glob(str(ROOT / "build" / "parts" / "*.py"))):
        txt = Path(f).read_text(encoding="utf8")
        chunks = re.split(r"^# %%(.*)$", txt, flags=re.M)
        # chunks: [pre, header1, body1, header2, body2 ...]
        for hdr, body in zip(chunks[1::2], chunks[2::2]):
            body = body.strip("\n")
            if not body.strip():
                continue
            if "[markdown]" in hdr:
                lines = [re.sub(r"^# ?", "", l) for l in body.split("\n")]
                cells.append(nbf.v4.new_markdown_cell("\n".join(lines)))
            else:
                cells.append(nbf.v4.new_code_cell(body))
    return cells

def main(execute=True):
    nb = nbf.v4.new_notebook()
    nb.cells = parse_parts()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / NB_NAME
    if execute:
        t = time.time()
        client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        print(f"executed in {time.time()-t:.0f}s")
    nbf.write(nb, out)
    print("wrote", out, "cells:", len(nb.cells))

if __name__ == "__main__":
    main(execute="--no-exec" not in sys.argv)
