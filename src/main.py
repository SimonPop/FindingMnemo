from WordGraph import WordGraph


def main():
    with open("../sherlock.txt") as f:
        text = f.read()
    wg = WordGraph(corpus=text, state_size=2)


    input_word = ''
    translation = ''

    closest_word = wg.find_closest_word(input_word)

    path = wg.find_path([closest_word, translation])

    print(path)

    # 1. Read config
    # 2. Create or Load Markov Chain
    # 3. Convert or Load Markov Chain into a Graph
    # 4. Input word to translate
    # 5. Translate input word
    # 6. Get primitives
    # 7. Find path though graph
    # 8. Create image based on path