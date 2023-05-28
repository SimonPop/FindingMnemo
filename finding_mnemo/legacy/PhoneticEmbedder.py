import eng_to_ipa as ipa
import panphon.distance
from dragonmapper import transcriptions


class PhoneticEmbedder:
    def __init__(self):
        self.dst = panphon.distance.Distance()

    def encode_pinyin(self, string: str) -> str:
        return transcriptions.pinyin_to_ipa(string)

    def encode_english(self, string: str) -> str:
        return ipa.convert(string)

    def naive_distance(self, str1: str, str2: str) -> float:
        return self.dst.levenshtein_distance(str1, str2)
