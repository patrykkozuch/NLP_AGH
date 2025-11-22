from speakleash import Speakleash

from config import DATASETS_DIR

sl = Speakleash(str(DATASETS_DIR))

datasets = [
    'plwikisource',
    'wolne_lektury_corpus',
    'plwiki',
    'web_artykuły_inne_147',
    'shopping_1_general_corpus',
    'forum_wizaz_pl_corpus',
    'web_artykuły_inne_148',
    'open_subtitles_corpus'
]

for dataset in datasets:
    print("Checking dataset:", dataset)
    if not sl.get(dataset).check_file()[0]:
        print(f"Downloaded {dataset} dataset.")
    else:
        print(f"{dataset} dataset already downloaded.")
