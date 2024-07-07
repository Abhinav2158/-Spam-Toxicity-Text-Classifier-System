import streamlit as st
import pickle
import string
pip install nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # Lower case
    text = nltk.word_tokenize(text)  # Tokenization

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)  # Removing special characters

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)  # Removing stopwords and punctuation

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Stemming

    return " ".join(y)

# Load the vectorizers and models
tfidf_spam = pickle.load(open('vectorizer.pkl', 'rb'))
model_spam = pickle.load(open('model.pkl', 'rb'))
tfidf_toxic = pickle.load(open('vectorizer_toxic_classifier.pkl', 'rb'))
model_toxic = pickle.load(open('model_toxic_classifier.pkl', 'rb'))

# Streamlit UI
st.title("Text Classification App")

# Sidebar for choosing the classifier
classifier = st.sidebar.selectbox("Choose the classifier", ("Spam Classifier", "Toxicity Classifier"))

if classifier == "Spam Classifier":
    st.header("Email/SMS Spam Classifier")
    input_sms = st.text_area("Enter your message")

    if st.button("Predict"):
        # Preprocess step
        transformed_sms = transform_text(input_sms)

        # Vectorize step
        vector_input = tfidf_spam.transform([transformed_sms])

        # Predict
        result = model_spam.predict(vector_input)[0]

        # Display
        if result == 1:
            st.header('Spam')
        else:
            st.header('Not Spam')

elif classifier == "Toxicity Classifier":
    st.header("Text Toxicity Classifier")
    input_text = st.text_area("Enter your message")

    if st.button("Predict"):
        # Preprocess step
        transformed_text = transform_text(input_text)

        # Vectorize step
        vector_input = tfidf_toxic.transform([transformed_text])

        # Predict
        result = model_toxic.predict(vector_input)[0]

        # Display
        if result == 1:
            st.header('Toxic')
        else:
            st.header('Not Toxic')
