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

    @requests(on=['/search', '/generate'])
    def search(self, docs: DocumentArray, **kwargs) -> Union[DocumentArray, Dict, None]:
        x = docs[:,'tags__ipa']
        docs.embeddings = self.model.encode(x).detach()

        # TODO: replace by self.da alone when DB is fixed.
        # ---
        # self.da = DocumentArray([Document(text='hotel_a', ipa='bɔ́təl'), Document(text='hotel_b', ipa='bɔ́təl')])
        # x = self.da[:,'tags__ipa']
        # self.da.embeddings = self.model.encode(x).detach()
        # ---

        docs.match(self.da, metric='cosine', limit=self.n_limit)
        return docs

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


if __name__ == '__main__':
    f = Flow().add(name='Engine', uses=Engine)
    with f:
        f.post(on='/generate', inputs=DocumentArray(Document(text='hotel', ipa='bɔ́təl')))