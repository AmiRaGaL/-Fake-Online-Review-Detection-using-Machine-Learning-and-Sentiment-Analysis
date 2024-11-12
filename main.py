import pickle
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# LOAD PICKLE FILES
model = pickle.load(open('Data/best_model.pkl', 'rb'))
vectorizer = pickle.load(open('Data/count_vectorizer.pkl', 'rb'))

# FOR STREAMLIT
#nltk.download('stopwords')

# TEXT PREPROCESSING
sw = set(stopwords.words('english'))


def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()

    cleaned = []
    stemmed = []

    for token in tokens:
        if token not in sw:
            cleaned.append(token)

    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)


def text_classification(text):
    if len(text) < 1:
        print("  ")
    else:
        cleaned_review = text_preprocessing(text)
        process = vectorizer.transform([cleaned_review]).toarray()
        prediction = model.predict(process)
        print(prediction)
        p = ''.join(str(i) for i in prediction)

        if p == 'True':
            print("The review entered is Legitimate.")
        if p == 'False':
            print("The review entered is Fraudulent.")

if __name__ == '__main__':
    print("Fraud Detection in Online Consumer Reviews Using Machine Learning Techniques")
    print("Model: Logistic Regression | Vectorizer: Count")
    review = input("Enter Review: ")
    if review!=None:
        text_classification(review)
