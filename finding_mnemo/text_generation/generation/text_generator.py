from typing import List

from docarray import DocumentArray
from keytotext import pipeline


class TextGenerator():
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {
            "max_length": 1024,
            "num_beams": 20,
            "length_penalty": 0.01,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
        self.model = pipeline("k2t-base")

    def generate(self, docs: List[str], **kwargs):
        # for query in docs:
        #     mnemo = [self.model([query.text, keyword], **self.config) for keyword in query.matches[:,'text']]
        #     query.tags['mnemo'] = mnemo
        # return docs
        return self.model(docs, **self.config)