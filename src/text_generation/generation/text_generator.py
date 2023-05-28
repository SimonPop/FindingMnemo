from typing import List

from jina import DocumentArray, Executor, requests
from keytotext import pipeline


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
        self.model = pipeline("k2t-base")

    @requests(on=["/generate"])
    def generate(self, docs: List[str], **kwargs):
        # for query in docs:
        #     mnemo = [self.model([query.text, keyword], **self.config) for keyword in query.matches[:,'text']]
        #     query.tags['mnemo'] = mnemo
        # return docs
        return self.model(docs, **self.config)


from jina import Document, Flow

if __name__ == "__main__":

    inputs = DocumentArray(
        [
            Document(text="football", matches=[Document(text="Zidane")]),
            Document(text="bakery", matches=[Document(text="bread")]),
        ]
    )

    f = Flow().add(name="TextGenerator", uses=TextGenerator)
    with f:
        f.post(on="/generate", inputs=inputs, on_done=print)

    # text_generator = TextGenerator()
    # text_generator.generate(inputs)
