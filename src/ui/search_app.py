import streamlit as st
import requests

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

chinese_word = st.text_input("Mandarin word:")

def display_candidates(response):
    matches = [match["text"] for x in response for match in x['matches'] ]
    scores = [match["scores"]["cosine"]["value"] for x in response for match in x['matches'] ]
    def score2emoji(score: float):
        if score < 0.01:
            return ":tropical_fish:"
        elif score < 0.05:
            return ":fish:"
        else:
            return ":blowfish:"
    emojis = [score2emoji(x) for x in scores]
    for index, (match, emoji) in enumerate(zip(matches, emojis)):
        st.write(f"{emoji} - {match} ")
    st.info(':tropical_fish: - Definitely Mnemo / :fish: - Not exactly Mnemo / :blowfish: - Barely Mnemo')

if chinese_word:
    response = requests.get(url=f"http://localhost:8000/search/{chinese_word}/").json()
    display_candidates(response)