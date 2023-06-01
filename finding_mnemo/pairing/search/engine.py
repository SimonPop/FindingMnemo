from pathlib import Path
from typing import Dict, Union

import torch
from docarray import Document, DocumentArray

from finding_mnemo.pairing.model.phonetic_siamese import PhoneticSiamese


class Engine():
    n_limit: int = 5
    model: PhoneticSiamese
    da: DocumentArray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.load_model()
        self.da = self.documents()

    def search(self, docs: DocumentArray, **kwargs) -> Union[DocumentArray, Dict, None]:
        x = docs[:, "tags__ipa"]
        docs.embeddings = self.model.encode(x).detach()
        docs.match(self.da, metric="euclidean", limit=self.n_limit)
        return docs

    def load_model(self) -> PhoneticSiamese:
        model = PhoneticSiamese()
        model.load_state_dict(
            torch.load(Path(__file__).parent.parent / "model" / "model_dict")
        )
        model.eval()
        self.model = model
        return model

    def load_documents(self) -> None:
        self.da = self.documents()

    def documents(self) -> DocumentArray:
        da = DocumentArray(
            storage="redis",
            config={
                "n_dim": self.model.embedding_dim,
                "index_name": "english_words",
                "distance": "L2",
                "host": "redis",
                "port": "6379",
            },
        )
        return DocumentArray(da, copy=True)