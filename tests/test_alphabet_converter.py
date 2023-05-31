from finding_mnemo.pairing.dataset.processing.alphabet_converter import AlphabetConverter

alphabet_converter = AlphabetConverter()

def test_convert_eng_to_ipa():
    ipa = alphabet_converter.convert_eng_to_ipa("mnemonic")
    assert ipa == "nɪmɑnɪk"

def test_convert_mandarin_to_ipa():
    ipa = alphabet_converter.convert_mandarin_to_ipa("Zhù jì cí")
    assert ipa == "ʈʂutɕitsʰɯ"

def test_convert_to_custom():
    custom = alphabet_converter.convert_to_custom("nɪmɑnɪk", True, 5)
    assert custom == "CACACAE"
    custom = alphabet_converter.convert_to_custom("nɪmɑnɪk", True, 10)
    assert custom == "CHGDCHE"
    custom = alphabet_converter.convert_to_custom("nɪmɑnɪk", True, 15)
    assert custom == "KBKFKBE"
    custom = alphabet_converter.convert_to_custom("nɪmɑnɪk", True, 20)
    assert custom == "ALIGALE"