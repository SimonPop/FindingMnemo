from pathlib import Path
from pydantic import BaseModel
from typing import List
from collections import deque
from bs4 import BeautifulSoup
import requests
from os.path import join
from src.dataset.wiktionary_graph.wiktionaryparser.core import WiktionaryParser

class WiktionaryEntry(BaseModel):
    title: str # Used as label
    definition: str # Later used to get an encoding.
    related_words: List[str] # Used to find neighbors in the graph

class Scrapper():
    """
    Scraps Wiktionary in order to create a training set for graph shortest path.
    """

    def __init__(self):
        self.url = 'https://www.wiktionary.org/wiki/'
        self.client = Neo4jClient(index_name="wiktionary")
        self.parser = WiktionaryParser()

    def breadth_first_search_scrapping(self, root: str, limit: int = None) -> List[WiktionaryEntry]:
        already_visited = self.client.list_words()
        queue = deque()
        self.scrap(root)
        while len(queue) > 0 and limit > 0:
            word = queue.pop()
            if not word in already_visited:
                entry = self.scrap(word)
                self.client.store(entry)
                queue.extend(entry.related_words)
                already_visited.append(word)
                limit -=1

    def scrap(self, word: str = "mnemonic") -> WiktionaryEntry:
        results = self.parser.fetch(word)
        links = []
        related_words = []
        for result in results:
            for definition in result["definitions"]:
                links.extend(definition['links'])
                related_words.extend(definition['relatedWords'])
            # result["pronunciations"] TODO pronunciations links
            # TODO definition / s
            # TODO: remove itself
        return WiktionaryEntry(title=word, definition="", related_words=links)

    def store_entry(self, entry: WiktionaryEntry) -> None:
        """Stores entry into Neo4J database.

        Args:
            entry (WiktionaryEntry): Entry to store.
        """
        pass

class Neo4jClient():
    def __init__(self, index_name: str):
        self.index_name = index_name

    def store(self, entry: WiktionaryEntry):
        pass

    def list_words(self) -> List[str]:
        pass

if __name__ == "__main__":
    from pprint import pprint
    scrapper = Scrapper()
    pprint(scrapper.scrap())