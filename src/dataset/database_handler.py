from neo4j import GraphDatabase
from src.dataset.wiktionary_graph.wiktionay_entry import WiktionaryEntry
from typing import List
import networkx as nx

class DatabaseHandler:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def store_entry(self, entry: WiktionaryEntry):
        self.create_node(entry.title, entry.definition)
        for related_entry in entry.related_words:
            self.create_relation(entry.title, related_entry)

    def create_node(self, title: str, definition: str):
        with self.driver.session() as session:
            session.run(f"CREATE (d:Definition {{title: '{title}', definition: '{definition}'}})")

    def create_relation(self, title_1: str, title_2: str):
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (d1:Definition {{title: '{title_1}'}})
                MERGE (d2:Definition {{title: '{title_2}'}})
                CREATE (d1)-[r:MENTIONS]->(d2)
                """)

    def load_graph(self) -> nx.Graph:
        return 

    def list_words(self) -> List[str]:
        with self.driver.session() as session:
            res = session.run(
                f"""
                MATCH (d:Definition)
                RETURN d
                """)
            return [x.data()['d']['title'] for x in res]
        return []

if __name__ == "__main__":
    entry = WiktionaryEntry(
        title="Test",
        definition="Test",
        related_words=[]
    )
    greeter = DatabaseHandler("bolt://localhost:7687", "simon", "wiktionary")
    greeter.store_entry(entry)
    greeter.close()