from dragonmapper import hanzi

def filter_chinese_ipa(ipa: str):
    from dragonmapper import transcriptions

    _IPA_CHARACTERS = transcriptions._IPA_CHARACTERS
    # Remove spaces, tones etc.
    return "".join([x for x in ipa if x in _IPA_CHARACTERS])


def convert_mandarin_to_ipa(h: str):
    try:
        return filter_chinese_ipa(hanzi.to_ipa(h))
    except:
        return "*"