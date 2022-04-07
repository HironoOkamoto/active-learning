import os

from tqdm import tqdm
from glob import glob

from sklearn.model_selection import train_test_split

orig_path = 'data/image'

for d in tqdm(sorted(glob(f"{orig_path}/*"))):
    f_names = sorted(glob(f"{d}/*"))
    if len(f_names):
        for tr in ["train", "test"]:
            os.makedirs(d.replace("image", tr), exist_ok=True)
        train_f_names, test_f_names = train_test_split(f_names, test_size=0.1, random_state=42)
        for f_name in train_f_names:
            os.system(f'mv {f_name} {d.replace("image", "train")}')
        for f_name in test_f_names:
            os.system(f'mv {f_name} {d.replace("image", "test")}')
    else:
        os.system(f'rm -rf {d}')
