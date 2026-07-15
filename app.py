# @Atul Kumar
# github-atul-kumar-30


from flask import Flask, render_template, request
import pandas as pd
import sklearn
import itertools
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup

app = Flask(__name__,template_folder='./templates',static_folder='./static')

loaded_model = pickle.load(open("model.pkl", 'rb'))
vector = pickle.load(open("vector.pkl", 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))
corpus = []

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text if text else None
    except:
        return None

def fake_news_det(news):
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = []
    for y in review :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))
    input_str = ' '.join(corpus)
    input_data = [input_str]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    
    try:
        feature_names = vector.get_feature_names_out()
    except AttributeError:
        feature_names = vector.get_feature_names()
        
    non_zero_indices = vectorized_input_data.nonzero()[1]
    
    word_contributions = []
    for idx in non_zero_indices:
        word = feature_names[idx]
        weight = loaded_model.coef_[0][idx] * vectorized_input_data[0, idx]
        word_contributions.append((word, weight))
        
    word_contributions.sort(key=lambda x: x[1], reverse=(prediction[0] == 1))
    top_words = [word for word, weight in word_contributions[:5]]
     
    return prediction, top_words     

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        input_type = request.form.get('inputType', 'text')
        message = request.form.get('news', '')
        
        if input_type == 'url':
            scraped_text = extract_text_from_url(message)
            if scraped_text:
                message = scraped_text
            else:
                return render_template('index.html', prediction_text="Failed to extract text from the provided URL. Please check the link or paste the text directly.")
                
        pred, top_words = fake_news_det(message)
        
        if pred[0] == 1:
            res = "Prediction of the News :  Looking Fake News📰"
            status = "fake"
        else:
            res = "Prediction of the News : Looking Real News📰 "
            status = "real"
            
        return render_template("index.html", prediction_text="{}".format(res), top_words=top_words, status=status)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)