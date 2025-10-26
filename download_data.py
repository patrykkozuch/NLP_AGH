from pathlib import Path
from speakleash import Speakleash

BASE_DIR = Path()
DATASETS_DIR = BASE_DIR / "datasets"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)


sl = Speakleash(str(DATASETS_DIR))

sl.get('1000_novels_corpus_CLARIN-PL').check_file()
sl.get('wolne_lektury_corpus').check_file()
sl.get('web_artykuły_inne_148').check_file()
sl.get('web_artykuły_inne_147').check_file()