from keytotext import pipeline
from typing import List
from jina import Executor, requests, DocumentArray

class TextGenerator(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {
            "max_length": 1024,
            "num_beams": 20,
            "length_penalty": 0.01,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
        self.model = pipeline('k2t-base')

    @requests(on=['/generate'])
    def generate(self, docs: DocumentArray, **kwargs):
        keywords = docs[:,'text']
        print('>>', self.model(keywords, **self.config))
        return None


from jina import Flow, Document

if __name__ == '__main__':
    f = Flow().add(name='TextGenerator', uses=TextGenerator)
    with f:
        f.post(on='/generate', inputs=DocumentArray([Document(text='football'), Document(text='Zidane')]), on_done=print)