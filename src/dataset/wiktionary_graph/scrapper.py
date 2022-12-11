from pathlib import Path
from pydantic import BaseModel
from typing import List

class WiktionaryEntry(BaseModel):
    title: str # Used as label
    definition: str # Later used to get an encoding.
    related_words: List[str] # Used to find neighbors in the graph

class Scrapper():
    """
    Scraps Wiktionary in order to create a training set for graph shortest path.
    """

    def __init__(self):
        self.url = Path('https://www.wiktionary.org/')

    def scrap(self, word: str) -> WiktionaryEntry:
        return WiktionaryEntry()

    def store_entry(self, entry: WiktionaryEntry) -> None:
        """Stores entry into Neo4J database.

        Args:
            entry (WiktionaryEntry): Entry to store.
        """
        pass