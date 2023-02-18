from fastapi import FastAPI
from jina import DocumentArray, Document

from src.text_generation.generation.text_generator import TextGenerator
from src.pairing.search.engine import Engine
from src.pairing.search.indexer import Indexer

from src.pairing.utils.ipa import convert_mandarin_to_ipa

from typing import List

app = FastAPI()

# Executors:
indexer = Indexer()
generator = TextGenerator()
engine = Engine()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/index/")
def index_items():
    documents = indexer.index()
    engine.load_documents()
    return f"Uploaded {len(documents)} documents."

@app.get("/search/{word}/")
def search(word: str):
    ipa: str = convert_mandarin_to_ipa(word)
    input = DocumentArray(Document(text=word, ipa=ipa))
    return engine.search(input).to_dict()

@app.get("/generate/{w1}/{w2}")
def generate(w1: str, w2: str):
    return generator.generate((w1, w2))