from dragonmapper import hanzi
import panphon
from dragonmapper import transcriptions

IPA_CHARS = [
    "s",
    "l",
    "ɤ",
    "d",
    "t",
    "ʃ",
    "ŋ",
    "z",
    "ʐ",
    "i",
    "n",
    "ɛ",
    "m",
    "r",
    "k",
    "ɻ",
    "θ",
    "ʈ",
    "ɨ",
    "ʊ",
    "e",
    "ɑ",
    "v",
    "a",
    "ɡ",
    "h",
    "ɥ",
    "p",
    "œ",
    "ɔ",
    "ʃ",
    "t",
    "æ",
    "d",
    "ʒ",
    "ɯ",
    "ɪ",
    "w",
    "o",
    "u",
    "ə",
    "j",
    "y",
    "b",
    "ɕ",
    "ɹ",
    "x",
    "ʒ",
    "ð",
    "ʂ",
    "f",
]

NON_IPA_CHARS = ["ʰ", "P", "I", "ʧ", "A"]

ft = panphon.FeatureTable()

IPA_FEATURE_DICT = {
            ipa: ft.word_to_vector_list(ipa, numeric=True)[0]
            for ipa in IPA_CHARS
    }

def filter_chinese_ipa(ipa: str) -> str:
    """Keeps only valid IPA character in a string of intended chinese IPA.
    Valid IPA characters are taken from dragonmapper's list of characters.

    Args:
        ipa (str): Input string of intended IPA characters.

    Returns:
        str: String with invalid IPA character removed.
    """
    from dragonmapper import transcriptions

    _IPA_CHARACTERS = transcriptions._IPA_CHARACTERS
    # Remove spaces, tones etc.
    return "".join([x for x in ipa if x in _IPA_CHARACTERS])


def convert_mandarin_to_ipa(h: str) -> str:
    """Converts a mandarin string to its IPA version.
    If a problem occurs in the translation, the result will be the default value: "*"

    Args:
        h (str): Input string to convert to IPA.

    Returns:
        str: IPA version of the input string or default value.
    """
    try:
        return filter_chinese_ipa(hanzi.to_ipa(h))
    except:
        return "*"
    
def mandarin_ipa_to_en(mandarin_str: str) -> str:
    repalcements = [("ʐ", "ʒ"),
    ("ɥ", "jʊa"),
    ('œ', 'ə'),
    ("ɻ", "ɪ"),
    ("x", "h"),
    ('ɨ', 'u'),
    ('ʂ', 'ʃ'),
    ('ʈ', 'd'),
    ('ɤ', 'ə'),
    ('w', 'w'),
    ('ɑ', 'a'),
    ('u', 'ə'),
    ('y', 'u'),
    ('ɯ', 'ʊ'),
    ('ɕ', 'ʃ')]

    for replacement in repalcements:
        mandarin_str = mandarin_str.replace(*replacement)

    return mandarin_str
