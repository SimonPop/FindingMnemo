import re

import wikipediaapi


class CorpusCreator:
    def __init__(self):
        self.wiki_wiki = wikipediaapi.Wikipedia("en")

    def create_corpus(self, page_name: str):
        page = self.wiki_wiki.page(page_name)
        text = page.text
        text = text.replace("\n", "")
        text = text.replace("'", "")
        with open(
            "corpus_{}.txt".format(page_name), "w", encoding="utf-8"
        ) as text_file:
            text_file.write("{}".format(text))
