from pathlib import Path
import json

class CustomPhonemes():
    def __init__(self):
        self.level_dict = {}
        for level in [5, 10, 15, 20]:
            f = open(Path(__file__).parent / 'phonemes_{}.json'.format(level), encoding='utf-8')
            data = json.load(f)
            f.close()
            self.level_dict[level] = data

    def convert(self, ipa: str, level: int = 10):
        custom_dict = self.level_dict[level]
        return ''.join([chr(65 + custom_dict[x]) if x in custom_dict else x for x in ipa])