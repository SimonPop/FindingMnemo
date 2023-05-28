from chinese_english_lookup import Dictionary
import streamlit as st
import requests
from dragonmapper import hanzi

title_alignment="""
<style>
h1 {
  text-align: center
}
.stAlert {
  text-align: center
}
</style>
"""
st.markdown(title_alignment, unsafe_allow_html=True)

st.title(":tropical_fish: Finding Mnemo")

d = Dictionary()
chinese_word = st.text_input("Mandarin word:") # 相机

def display_candidates(response, translation, pinyin):
    matches = [match["text"] for x in response for match in x['matches'] ]
    scores = [match["scores"]["euclidean"]["value"] for x in response for match in x['matches'] ]
    def score2emoji(score: float):
        if score < 0.01:
            return ":tropical_fish:"
        elif score < 0.05:
            return ":fish:"
        else:
            return ":blowfish:"
    emojis = [score2emoji(x) for x in scores]
    for index, (match, emoji) in enumerate(zip(matches, emojis)):
        generation = requests.get(url=f"http://fastapi:8000/generate/{translation}/{match}/").json()
        st.write(f"{emoji} - {match} - {generation}")
    st.info(':tropical_fish: - Definitely Mnemo / :fish: - Not exactly Mnemo / :blowfish: - Barely Mnemo')

if chinese_word:
    pinyin = hanzi.to_pinyin(chinese_word)
    response = requests.get(url=f"http://fastapi:8000/search/{pinyin}/").json()
    translation = d.lookup(chinese_word).definition_entries[0].definitions[0].split('(')[0]
    st.write(f"pinyin: {pinyin}")
    st.write(f"translation: {translation}")
    display_candidates(response, translation, pinyin)