from pydantic import BaseModel

class Pair(BaseModel):
    word_a: str
    word_b: str
    ipa_a: str
    ipa_b: str
    distance: float

class InterlanguagePair(BaseModel):
    chinese_hanzi: str
    chinese_pinyin: str
    chinese_ipa: str
    english_word: str
    english_ipa: str
    distance: float