from model.phonetic_siamese import PhoneticSiamese

from jina import requests, DocumentArray, Document, Executor, Flow
from typing import Dict, Union
from pathlib import Path
import torch

class Engine(Executor):
    n_limit: int = 5
    model: PhoneticSiamese
    da: DocumentArray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.load_model()
        self.da = self.documents()

    @requests(on=['/search', '/generate'])
    def search(self, docs: DocumentArray, **kwargs) -> Union[DocumentArray, Dict, None]:
        x = docs[:,'tags__ipa']
        docs.embeddings = self.model.encode(x).detach()
        docs.match(self.da, metric='cosine', limit=self.n_limit)
        return docs

    def load_model(self) -> PhoneticSiamese:
        model = PhoneticSiamese()
        model.load_state_dict(torch.load(Path(__file__).parent.parent / "model" / "model_dict"))
        model.eval()
        self.model = model
        return model

    def load_documents(self) -> None:
        self.da = self.documents()

    def documents(self) -> DocumentArray:
        da = DocumentArray(
            storage='redis',
            config={
                'n_dim': self.model.embedding_dim,
                'index_name': 'english_words',
                'distance': 'COSINE',
                'host': 'redis',
                'port': '6379',
            }
        )
        return DocumentArray(da, copy=True)


if __name__ == '__main__':
    f = Flow().add(name='Engine', uses=Engine)
    with f:
        f.post(on='/generate', inputs=DocumentArray(Document(text='hotel', ipa='bɔ́təl')))