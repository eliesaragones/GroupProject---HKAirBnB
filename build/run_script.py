"""Fast dev runner: concatenates the code cells of build/parts/*.py and runs them as one script (Agg backend)."""
import sys, re, glob, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import os; os.chdir(ROOT)
import matplotlib; matplotlib.use("Agg")
import builtins
builtins.display=lambda x: print(x)
code = []
for f in sorted(glob.glob(str(ROOT / "build" / "parts" / "*.py"))):
    txt = Path(f).read_text(encoding="utf8")
    chunks = re.split(r"^# %%(.*)$", txt, flags=re.M)
    for hdr, body in zip(chunks[1::2], chunks[2::2]):
        if "[markdown]" not in hdr:
            code.append(body)
src = "\n".join(code)
upto = sys.argv[1] if len(sys.argv) > 1 else None
if upto:
    src = src.split(f"#STOP:{upto}")[0]
t = time.time()
exec(compile(src, "notebook", "exec"), {"__name__": "__main__"})
print(f"--- done in {time.time()-t:.0f}s")
