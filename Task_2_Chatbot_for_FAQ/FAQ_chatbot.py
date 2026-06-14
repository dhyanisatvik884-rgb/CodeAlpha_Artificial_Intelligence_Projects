import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_data = {
    "what is the name of your country":"my country's name is india",
    "what is the capital of your country":"my country's capital is new delhi",
    "how many states are there in your country":"there are total 28 states in my country",
    "what is the national animal of your country":"the national animal of my country is the bengal tiger",
    "what is the national bird of your country":"the national bird of my country is the indian peafowl",
    "what is the national flower of your country":"the national flower of my country is the lotus",
    "what is the national fruit of your country":"the national fruit of my country is the mango",
    "what is the national tree of your country":"the national tree of my country is the banyan tree",
    "what is the national river of your country":"the national river of my country is the ganga",
    "what is the national anthem of your country":"the national anthem of my country is jana gana mana",
    "what is the national song of your country":"the national song of my country is vande mataram",
    "who is the president of your country":"the president of my country is draupadi murmu",
    "who is the prime minister of your country":"the prime minister of my country is narendra modi",
    "which is the largest state in your country":"rajasthan is the largest state in my country by area",
    "which is the smallest state in your country":"goa is the smallest state in my country by area",
    "which is the most populous state in your country":"uttar pradesh is the most populous state in my country",
    "what is the currency of your country":"the currency of my country is the indian rupee",
    "what is the currency symbol of your country":"the currency symbol of my country is ₹",
    "which languages are widely spoken in your country":"many languages are spoken in my country including hindi and english",
    "what is the largest city in your country":"mumbai is the largest city in my country by population",
    "when did your country gain independence":"my country gained independence on 15 august 1947",
    "who is known as the father of the nation":"mahatma gandhi is known as the father of the nation",
    "what is the national sport of your country":"india does not have an officially declared national sport",
    "which is the highest mountain peak in your country":"kangchenjunga is the highest mountain peak in my country",
    "which ocean lies south of your country":"the indian ocean lies south of my country",
    "what are the colors of your national flag":"the colors of my national flag are saffron white and green",
    "how many union territories are there in your country":"there are 8 union territories in my country",
    "which festival is widely celebrated in your country":"many festivals are celebrated in my country including diwali and holi",
    "what is the national calendar of your country":"the national calendar of my country is the saka calendar"
}

faq_ques = list(faq_data.keys())
vectorizer = TfidfVectorizer(stop_words='english')

faq_vectors = vectorizer.fit_transform(faq_ques)

st.title("FAQ Chatbot: Indian General Knowledge")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("What is your question???")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    user_vector = vectorizer.transform([user_input])
    similar_score = cosine_similarity(user_vector, faq_vectors)
    idx_match = similar_score.argmax()
    highscore = similar_score[0][idx_match]
    
    if highscore > 0.2:
        ques_find = faq_ques[idx_match]
        bot_reply = faq_data[ques_find]
    else:
        bot_reply = "I am sorry, I don't understand that question."
        
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})