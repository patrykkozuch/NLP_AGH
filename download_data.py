from pathlib import Path
from speakleash import Speakleash

BASE_DIR = Path()
DATASETS_DIR = BASE_DIR / "speakleash_dataset"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)


sl = Speakleash(str(DATASETS_DIR))

sl.get('plwikisource').check_file()
sl.get('wolne_lektury_corpus').check_file()
