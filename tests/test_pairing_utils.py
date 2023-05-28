import pytest
from finding_mnemo.pairing.utils.ipa import filter_chinese_ipa, convert_mandarin_to_ipa

def test_invalid_mandarin_ipa():
    invalid_mandarin = "000.:;"
    invalid_translation = convert_mandarin_to_ipa(invalid_mandarin)
    assert invalid_translation == ""

def test_valid_mandarin_ipa():
    valid_mandarin = "píngguŏ"
    valid_translation = convert_mandarin_to_ipa(valid_mandarin)
    assert valid_translation == "pʰiŋku"

def test_filtering_ipa():
    from dragonmapper import transcriptions
    _IPA_CHARACTERS = transcriptions._IPA_CHARACTERS
    valid_translation = filter_chinese_ipa(_IPA_CHARACTERS + "_-10=+")
    assert valid_translation == _IPA_CHARACTERS