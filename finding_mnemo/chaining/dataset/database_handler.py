from typing import List

import networkx as nx
from neo4j import GraphDatabase

from finding_mnemo.chaining.dataset.wiktionary_graph.wiktionay_entry import \
    WiktionaryEntry


class DatabaseHandler:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def store_entry(self, entry: WiktionaryEntry):
        self.create_or_set_node(entry.title, entry.definition)
        for related_entry in entry.related_words:
            self.create_relation(entry.title, related_entry)

    def create_or_set_node(self, title: str, definition: str):
        title = (
            title.replace("'", " ")
            .replace("{", " ")
            .replace("}", " ")
            .replace('"', " ")
            .replace("\\", " ")
        )
        definition = (
            definition.replace("'", " ")
            .replace("{", " ")
            .replace("}", " ")
            .replace('"', " ")
            .replace("\\", " ")
        )
        with self.driver.session() as session:
            query = f""" MERGE (d:Definition {{title: "{title}"}})
                ON CREATE SET d.definition = "{definition}"
                ON MATCH SET d.definition = "{definition}"
            """
            session.run(query)

    def create_relation(self, title_1: str, title_2: str):
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (d1:Definition {{title: '{title_1}'}})
                MERGE (d2:Definition {{title: '{title_2}'}})
                CREATE (d1)-[r:MENTIONS]->(d2)
                """
            )

    def load_graph(self) -> nx.Graph:
        query = f"""
        MATCH (d1:Definition)-[r:MENTIONS]->(d2:Definition)
        RETURN *
        """
        graph = nx.Graph()
        with self.driver.session() as session:
            result = session.run(query)
            data = result.data()
        for link in data:
            a, _, b = link["r"]
            graph.add_edge(a["title"], b["title"])
        return graph

    def list_words(self) -> List[str]:
        with self.driver.session() as session:
            res = session.run(
                f"""
                MATCH (d:Definition)
                RETURN d
                """
            )
            return [x.data()["d"]["title"] for x in res]
        return []


if __name__ == "__main__":
    # entry = WiktionaryEntry(
    #     title="Test",
    #     definition="Test",
    #     related_words=[]
    # )
    # greeter = DatabaseHandler("bolt://localhost:7687", "simon", "wiktionary")
    # greeter.store_entry(entry)
    # greeter.close()
    handler = DatabaseHandler("bolt://localhost:7687", "simon", "wiktionary")
    res = handler.load_graph()
    print(res.nodes(data=True))
