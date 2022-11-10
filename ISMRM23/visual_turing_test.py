import os
import numpy as np
import shutil
import pandas as pd

real_path = "/raid/home/follels/Documents/latent-diffusion/samples/real"
fake_path = "/raid/home/follels/Documents/latent-diffusion/samples/fake"
target_path = "/raid/home/follels/Documents/latent-diffusion/samples/likert_scale_test"

np.random.seed(42)

df = pd.DataFrame(columns=["real_path", "fake_path"])

for i in range(100):
    real = os.path.join(real_path, f"sample_{i}.png")
    fake = os.path.join(fake_path, f"sample_{i}.png")
    order = np.random.randint(2)
    if order == 0:
        target_real = os.path.join(target_path, f"sample_{i * 2}.png")
        target_fake = os.path.join(target_path, f"sample_{i * 2 + 1}.png")
    else:
        target_real = os.path.join(target_path, f"sample_{i * 2 + 1}.png")
        target_fake = os.path.join(target_path, f"sample_{i * 2}.png")

    shutil.copyfile(real, target_real)
    shutil.copyfile(fake, target_fake)
    
    new_row = pd.Series({"real_path": target_real, "fake_path": target_fake})
    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

df.to_csv(os.path.join(os.path.dirname(target_path), "overview.csv"), index=False)
    

