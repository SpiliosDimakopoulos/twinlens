import sys, time
sys.path.insert(0, ".")
import pandas as pd
from twinlens.core.gpt2 import GPT2
from twinlens.twin.replacement import Twin, TWIN_LEVELS
from twinlens.fidelity.engine import run_prompt
twin = Twin(GPT2.load())
rows = []
t = time.perf_counter()
for i, p in enumerate(["The Colosseum is located in the city of", "When Mary and John went to the store, John gave a drink to", "def add(a, b):\n    return"]):
    rows += run_prompt(twin, p, "x", i, seed=i)
print(f"{len(rows)} edits in {time.perf_counter()-t:.0f}s")
df = pd.DataFrame(rows)
for mode in TWIN_LEVELS:
    sp = df[[f"{mode}_dlogp", "real_dlogp"]].corr(method="spearman").iloc[0, 1]
    err = (df[f"{mode}_dlogp"] - df.real_dlogp).abs()
    tol = (err <= 0.1 + 0.25 * df.real_dlogp.abs()).mean()
    big = df.real_dlogp.abs() > 0.05
    sign = ((df[f"{mode}_dlogp"] > 0) == (df.real_dlogp > 0))[big].mean()
    cat = (err > 5).mean()
    print(f"{mode:17s} spearman={sp:.3f} within-tol={tol:.2%} sign-agree(|real|>.05)={sign:.2%} catastrophic(|err|>5)={cat:.2%} medcos={df[f'{mode}_cos'].median():.3f}")
