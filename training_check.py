import json
from pathlib import Path
from statistics import mean

files = [f"./results/results3/MMLU_ID_{i}.json" for i in range(5)]
runs = [json.loads(Path(p).read_text()) for p in files]

n_q = {len(r) for r in runs}    # sanity check
assert len(n_q) == 1, "runs differ in length"
n_q = n_q.pop()

good = 0
for idx in range(n_q):
    ref = runs[0][idx]["reference"]
    hits = sum(r[idx]["predicted"] == ref for r in runs)
    if hits >= 4:
        good += 1

acc = good / n_q * 100
print(f"questions: {n_q}")
print(f"consensus correct: {good}")
print(f"accuracy: {acc:.2f}%")
