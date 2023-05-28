from dragonmapper import hanzi


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
