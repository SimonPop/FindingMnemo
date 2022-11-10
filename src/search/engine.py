from src.model.sound_siamese_v2 import SoundSiamese

from jina import requests, DocumentArray, Document, Executor, Flow
from typing import Dict, Union
from pathlib import Path
import torch

class Engine(Executor):
    n_limit: int = 5
    model: SoundSiamese
    da: DocumentArray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.load_model()
        self.da = self.documents()

    @requests(on=['/search'])
    def search(self, docs: DocumentArray, **kwargs) -> Union[DocumentArray, Dict, None]:
        x = docs[:,'tags__ipa']
        np_query = self.model.encode(x).detach().numpy()[0]
        return self.da.find(np_query, limit=self.n_limit)

    def load_model(self) -> SoundSiamese:
        model = SoundSiamese()
        model.load_state_dict(torch.load(Path(__file__).parent.parent / "model" / "model_dict"))
        self.model = model
        return model

    def documents(self) -> DocumentArray:
        return DocumentArray(
            storage='redis',
            config={
                'n_dim': self.model.embedding_dim,
                'index_name': 'english_words',
                'distance': 'COSINE'
            },
        )


# f = Flow().add(uses=Engine)

# with f:
#     f.post(on='/search', inputs=DocumentArray(Document(text='hotel', ipa='bɔ́təl')))


# m = Engine()
# da = DocumentArray([Document(text='hotel', ipa='bɔ́təl')])
# res = m.search(da)
# print(res[:,'text'])
# print(res.summary())