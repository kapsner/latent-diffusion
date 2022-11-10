import pandas as pd 
import seaborn as sns
sns.set_context("paper", font_scale=1.25)
import matplotlib.pyplot as plt
import numpy as np

import os

def extract_id(cases):
    ids = [int(os.path.basename(c).split("_")[1].split(".")[0]) for c in cases]
    return ids


df = pd.read_csv("/raid/home/follels/Documents/latent-diffusion/samples/overview.csv")
real_cases = extract_id(df["real_path"].tolist())
fake_cases = extract_id(df["fake_path"].tolist())

results = {
    0: extract_id(os.listdir("/raid/home/follels/Documents/latent-diffusion/samples/sebastians_reading/0 - unrealistic")),
    1: extract_id(os.listdir("/raid/home/follels/Documents/latent-diffusion/samples/sebastians_reading/1 - slightly unrealistic")),
    2: extract_id(os.listdir("/raid/home/follels/Documents/latent-diffusion/samples/sebastians_reading/2 - indeterminate")),
    3: extract_id(os.listdir("/raid/home/follels/Documents/latent-diffusion/samples/sebastians_reading/3 - quite realistic")),
    4: extract_id(os.listdir("/raid/home/follels/Documents/latent-diffusion/samples/sebastians_reading/4 - fully realistic")),
}

fig, ax = plt.subplots()
for name, cases in zip(["Real", "Fake"], [real_cases, fake_cases]):
    print(f"{name} evaluation:")
    result_hist = []
    for c in range(5):
        result = len([r for r in results[c] if r in cases])
        result_hist.append(result)
        print(f"Class {c}: {result}.")
    result_hist = np.array(result_hist)
    print(result_hist)
    delta = 0 if name == "Real" else 0.3
    bar = plt.bar(np.arange(5) + delta, result_hist, 0.3, label=name, color="#012A36" if name == "Real" else "#85BDBF")
    ax.bar_label(bar, padding=3)
    

plt.xlabel('Likert Score')
plt.ylabel('Number of cases')
plt.xticks(np.arange(5) + 0.3 / 2, ("0", "1", "2", "3", "4"))
plt.legend(loc='best')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join("/raid/home/follels/Documents/latent-diffusion/ISMRM23", f"Histogram.png"), dpi=400)
