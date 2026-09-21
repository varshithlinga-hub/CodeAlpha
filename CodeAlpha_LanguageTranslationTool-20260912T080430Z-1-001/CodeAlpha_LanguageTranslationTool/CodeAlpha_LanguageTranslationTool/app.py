import streamlit as st
from deep_translator import GoogleTranslator

st.title("Language Translation Tool")

text = st.text_area("Enter text to translate")

source_lang = st.selectbox(
    "Source Language",
    ["auto", "english", "hindi", "telugu", "tamil", "spanish", "french"]
)

target_lang = st.selectbox(
    "Target Language",
    ["english", "hindi", "telugu", "tamil", "spanish", "french"]
)

if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        st.subheader("Translated Text")
        st.write(translated)