import streamlit as st
import requests

st.title("Pair Search App")

chinese_word = st.text_input("chinese_word")

if chinese_word:
    response = requests.get(url=f"http://localhost:8000/search/{chinese_word}/")
    st.write(response.json())