from jina import Flow, Document, DocumentArray
from src.generation.text_generator import TextGenerator
from src.search.engine import Engine
from src.search.indexer import Indexer


if __name__ == "__main__":
    f = Flow().add(name='Indexer', uses=Indexer).add(name='Engine', uses=Engine, needs=[]).add(name='TextGenerator', uses=TextGenerator, needs=['Engine'])
    
    f.to_docker_compose_yaml('flow-docker-compose.yml')
    
    # with f:
    #     f.post(on='/generate', inputs=DocumentArray(Document(text='hotel', ipa='bɔ́təl')), on_done=print)

    # inputs = DocumentArray([Document(text='hotel', ipa='bɔ́təl')])
    # x = engine = Engine().search(inputs)
    # x = TextGenerator().generate(x)
    # print(x[:,'tags__mnemo'])
    # print(x[:,'text'])
    # print(x['@m','text'])