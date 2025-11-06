from speakleash import Speakleash

from config import DATASETS_DIR

sl = Speakleash(str(DATASETS_DIR))

datasets = [
    'plwikisource',
    'wolne_lektury_corpus',
]

for dataset in datasets:
    print("Checking dataset:", dataset)
    if not sl.get(dataset).check_file()[0]:
        print(f"Downloaded {dataset} dataset.")
    else:
        print(f"{dataset} dataset already downloaded.")
