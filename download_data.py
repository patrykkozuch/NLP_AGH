from speakleash import Speakleash

from config import DATASETS_DIR

sl = Speakleash(str(DATASETS_DIR))

sl.get('plwikisource').check_file()
sl.get('wolne_lektury_corpus').check_file()
