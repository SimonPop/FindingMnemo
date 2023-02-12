# Finding 🐠 Mnemo 

This repo contains a prototype for a mnemotechnic translation generator tool.

For a given word you want to remember, finds the closest phonetic match to a known word and a link chain of ideas that allows to go from one to the other.

__Example__: 

The word `努力` is pronounced `nǔ lì`. It means `effort`. The closest sounding word found in our english vocabulary would be `newly`. We then want to create a chain or sentence that links both words such as: "*`Newly` selected players have to make an `effort` to compete with long established ones.*"
 

![](imgs/flowchart.jpg)


## User Guide

### Search Engine

The search engine represents the first component of this prototype.
It gives access to the english words that sound the closest from an input mandarin word.

#### Deployment

You need to deploy the application in containers. 
This app requires both a Database (Redis) and a backend (FastAPI).
In order to deploy both, we would use the `docker-compose.yaml` file using the command:
```bash
docker compose up 
```

That should pop two docker containers, which you can check on the UI or using `docker ps`.

![](imgs/container-screen.PNG)
*Docker UI after using compose-up command*

## Technical Documentation

Three components are currently in development in this prototype:
- __Pairing component__: Finding an english word sounding like a given mandarin word.
- __Keyphrase component__: Generates a sentence mixing both paired words: translation of the mandarin word and sound alike word.
- __Chaining component__: Finds a chain of words connecting paired words.

### Pairing

In order to find a word that sounds like another, we want to use [levenshtein's distance](https://en.wikipedia.org/wiki/Levenshtein_distance).

However this distance, is very computation heavy and takes prohibitive time to run over a large number of words. 

Instead, we will learn a proxy model that produces an embedding space where distances are equivalent to levenshtein's distance.

This way, we can rely on neural search techniques in order to pre-process all english words, and only have to process input words an match them efficiently. 

#### Pairing model

This model relies on a transformer architecture applied on [IPA characters](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet). 

We use it to encode words to the embedding space discussed above and then use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) in order to evaluate the phonetic distance between them.

![](imgs/parallel_plot_pairing.PNG)
*Parallel plot of model's performance in contrast with some hyperparameters*

#### Dataset

It is trained with Triplet Loss, on a dataset generated using the original levenshtein's distance (Weak supervision).

Triplet loss works using a *margin* value, separating the anchor word and positive word from the negative word. We intent to use the knowledge of the exact distance to generate words that fit exactly or as close as possible from this margin (i.e. most relevant pairs of words).

### Key-chain

In order to find a connection between two words, a first approach is to link them via a knowledge graph: find a path in that graph that links both of these words. 

#### Wikipedia roaming

We use Wikipedia as our knowledge graph: each page is a node, and link to other pages represent edges. (Wiktionary has been tried but seems less complete).

![](imgs/path_finding.jpg)
*Finding path from "Finding Nemo" to "China" through Wikipedia knowledge graph*

The fullest approach using Wikipedia as our graph would be to find one of the shortest path from a word to another. 

#### Shortest path model

In order to reduce the number of call we make to Wikipedia's API, we are looking to find the best pages to explore. That can be achieved using A* algorithm with some heuristic for distance estimation.

Machine learning model can be used to predict how far a page is from another and be used inside A* as the heuristic.

A Graph Neural Network as well as a standard ML approach have been tried.

However API calls still make the process very slow.


#### Link prediction

A lesser approach, but faster one is simply to start with a finite graph (e.g. a graph for all english words of our vocabulary) and simply find the word in the graph our input word is the closest from, and then compute the path in that graph to our final word instead of roaming Wikipedia to find the optimal path.

![](imgs/link_prediction.jpg)
*Finding path from "Elephant" to "China" through restricted Wikipedia knowledge graph*

#### Link prediction model

[TODO]

### Keyphrase

Instead of finding a chain of concepts in Wikipedia, we could try to generate key sentences to remember.

In order to do that, a model from HuggingFace: [key-to-text](https://huggingface.co/gagan3012/k2t-base), has been tried, but seems very far from an acceptable result.


