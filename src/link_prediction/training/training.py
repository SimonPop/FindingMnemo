import networkx as nx
from src.link_prediction.dataset.create_classic_dataset import create_dataset
from sklearn.ensemble import RandomForestClassifier
import json

def train():

    model = RandomForestClassifier()

    with open('graph.json', 'r') as f:
        G = nx.node_link_graph(json.load(f))
    dataset = create_dataset(G, 100)
    features = ["TF-IDF",  "common_neighbors",  "common_categories",  "degree_sum",  "degree_diff"]
    X_train, y_train = dataset['train'][features], dataset['train']["connected"]
    X_val, y_val = dataset['val'][features], dataset['val']["connected"]
    X_test, y_test = dataset['test'][features], dataset['test']["connected"]

    model.fit(X_train, y_train)
    model.score(X_val, y_val)

if __name__ == '__main__':
    train()