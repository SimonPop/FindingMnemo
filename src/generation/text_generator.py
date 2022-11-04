from keytotext import pipeline
from typing import List

class TextGenerator():
    def __init__(self):
        self.config = {
            "max_length": 1024,
            "num_beams": 20,
            "length_penalty": 0.01,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
        self.model = pipeline('k2t-base')

    def generate(self, keywords: List[str]):
        return self.model(keywords, **self.config)