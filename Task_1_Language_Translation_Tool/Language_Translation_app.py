import streamlit as st
from deep_translator import GoogleTranslator

st.title("LANGUAGE TRANSLATION TOOL")

text_by_user = st.text_area("Enter the text to translate:\n\t")
languages = GoogleTranslator().get_supported_languages()

src_lang = st.selectbox("\nSelect Source Language:\n",languages)
dest_lang = st.selectbox("\nSelect Target Language:\n",languages)

if st.button("Translate"):
    if text_by_user.strip():
        translator = GoogleTranslator(source=src_lang, target=dest_lang)
        op_text = translator.translate(text_by_user)
        st.success(op_text)
