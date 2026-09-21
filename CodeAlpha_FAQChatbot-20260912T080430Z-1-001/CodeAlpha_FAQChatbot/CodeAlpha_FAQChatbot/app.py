import streamlit as st
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = {
    "What is artificial intelligence?": "Artificial Intelligence is the ability of machines to perform tasks that normally require human intelligence.",
    "What is machine learning?": "Machine Learning is a branch of AI where computers learn from data and improve automatically.",
    "What is deep learning?": "Deep Learning uses neural networks to learn patterns from large amounts of data.",
    "What is NLP?": "NLP stands for Natural Language Processing. It helps computers understand human language.",
    "What is computer vision?": "Computer Vision helps computers understand and process images and videos.",
    "What is a chatbot?": "A chatbot is a software program that can answer user questions automatically.",
    "What is cosine similarity?": "Cosine similarity is used to measure how similar two pieces of text are.",
    "What is Streamlit?": "Streamlit is a Python library used to create simple web apps.",
    "How does this chatbot work?": "This chatbot compares your question with stored FAQ questions and gives the most similar answer.",
    "Which technologies are used in this project?": "This project uses Python, Streamlit, NLTK, TF-IDF, and cosine similarity."
}

questions = list(faqs.keys())
answers = list(faqs.values())

def preprocess_text(text):
    tokens = wordpunct_tokenize(text.lower())
    clean_tokens = [word for word in tokens if word.isalpha()]
    return " ".join(clean_tokens)

processed_questions = [preprocess_text(question) for question in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

st.title("FAQ Chatbot")
st.write("Ask me questions related to Artificial Intelligence.")

user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        processed_user_question = preprocess_text(user_question)
        user_vector = vectorizer.transform([processed_user_question])

        similarities = cosine_similarity(user_vector, question_vectors)
        best_match_index = similarities.argmax()
        best_score = similarities[0][best_match_index]

        st.subheader("Bot Response")

        if best_score >= 0.2:
            st.success(answers[best_match_index])
            st.write("Matched FAQ:", questions[best_match_index])
        else:
            st.error("Sorry, I could not find a good answer. Please ask in a different way.")

st.markdown("---")
st.subheader("Sample Questions")

for question in questions:
    st.write("- " + question)