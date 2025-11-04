from speakleash import Speakleash

from config import DATASETS_DIR

sl = Speakleash(str(DATASETS_DIR))

datasets = [
    'plwikisource',
    '1000_novels_corpus_CLARIN-PL',
    'plwiki',
    'wolne_lektury_corpus',
    'web_artykuły_inne_148',
]

for dataset in datasets:
    print("Checking dataset:", dataset)
    if not sl.get(dataset).check_file()[0]:
        print(f"Downloaded {dataset} dataset.")
    else:
        print(f"{dataset} dataset already downloaded.")
