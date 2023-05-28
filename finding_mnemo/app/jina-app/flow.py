from jina import Document, DocumentArray, Flow

from finding_mnemo.pairing.search.engine import Engine
from finding_mnemo.pairing.search.indexer import Indexer
from text_generation.generation.text_generator import TextGenerator

if __name__ == "__main__":
    f = (
        Flow()
        .add(name="Indexer", uses=Indexer)
        .add(name="Engine", uses=Engine, needs=[])
        .add(name="TextGenerator", uses=TextGenerator, needs=["Engine"])
    )

    f.to_docker_compose_yaml("flow-docker-compose.yml")

    # with f:
    #     f.post(on='/generate', inputs=DocumentArray(Document(text='hotel', ipa='bɔ́təl')), on_done=print)

    # inputs = DocumentArray([Document(text='hotel', ipa='bɔ́təl')])
    # x = engine = Engine().search(inputs)
    # x = TextGenerator().generate(x)
    # print(x[:,'tags__mnemo'])
    # print(x[:,'text'])
    # print(x['@m','text'])
