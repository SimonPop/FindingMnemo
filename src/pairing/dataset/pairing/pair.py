from pydantic import BaseModel

class Pair(BaseModel):
    chinese_hanzi: str
    chinese_pinyin: str
    chinese_ipa: str
    english_word: str
    english_ipa: str
    distance: float