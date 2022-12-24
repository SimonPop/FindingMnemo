from typing import List
from collections import deque
from src.dataset.wiktionary_graph.wiktionaryparser.core import WiktionaryParser
from src.dataset.wiktionary_graph.wiktionay_entry import WiktionaryEntry
from dataset.database_handler import DatabaseHandler

class Scrapper():
    """
    Scraps Wiktionary in order to create a training set for graph shortest path.
    """

    def __init__(self):
        self.url = 'https://www.wiktionary.org/wiki/'
        self.client = DatabaseHandler(uri="bolt://localhost:7687", user="simon", password="wiktionary")
        self.parser = WiktionaryParser()

    def breadth_first_search_scrapping(self, root: str, limit: int = None) -> List[WiktionaryEntry]:
        already_visited = self.client.list_words()
        def handle_word(word):
            entry = self.scrap(word)
            self.client.store_entry(entry)
            queue.extend(entry.related_words)
            already_visited.append(word)
        queue = deque()
        handle_word(root)
        while len(queue) > 0 and limit > 0:
            word = queue.pop()
            if not word in already_visited:
                handle_word(word)
                limit -=1
        return 

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
        self.client.store_entry(entry)

# class Neo4jClient():
#     def __init__(self, index_name: str):
#         self.index_name = index_name
#         self.temporary_store = []

#     def store(self, entry: WiktionaryEntry):
#         # TODO: use Neo4J instead.
#         self.temporary_store.append(entry)

#     def list_words(self) -> List[str]:
#         # TODO: use Neo4J instead.
#         return [e.title for e in self.temporary_store]

if __name__ == "__main__":
    scrapper = Scrapper()
    scrapper.breadth_first_search_scrapping("mnemonic", 200)