from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import string
import regex as re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)
model_a = pickle.load(open("model/a_classifier.pkl", "rb"))
model_sa = pickle.load(open("model/sa_classifier.pkl", "rb"))
model_ss = pickle.load(open("model/ss_classifier.pkl", "rb"))
tfidf_a = pickle.load(open("model/a_vectorizer.pkl", "rb"))
tfidf_sa = pickle.load(open("model/sa_vectorizer.pkl", "rb"))
tfidf_ss = pickle.load(open("model/ss_vectorizer.pkl", "rb"))

kamus = pd.read_csv("data/preprocessing/words.csv")
file_sw = open("data/preprocessing/stopwordbahasa.csv", "r")
kata_normalisasi_dict = {}
sw = []
factory = StemmerFactory()
stemmer = factory.create_stemmer()
remove_lists = ["nya", "pun", "yang", "an"]
exceptions_sw = ["tidak", "jangan", "guna", "manfaat"]
for index,row in kamus.iterrows():
    if row[0] not in kata_normalisasi_dict:
        kata_normalisasi_dict[row[0]] = row[1]
for line in file_sw:
    stripped_line = line.strip()
    sw.append(stripped_line)
for ex in exceptions_sw:
    if ex in sw:
        sw.remove(ex)
        
def vectorize(i_data,i_tfidf_vect_fit):
    X_tfidf = i_tfidf_vect_fit.transform(i_data)
    words = i_tfidf_vect_fit.get_feature_names()
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    X_tfidf_df.columns = words
    return(X_tfidf_df)
def punc_casefold_lower_token(sentence):
    str_rm_punctuation = \
        sentence.translate(str.maketrans("", "", string.punctuation))
    str_no_number = re.sub('[^a-zA-Z]', ' ', str_rm_punctuation)
    return str_no_number.lower().split()
def pp_spelling(document):
    return [kata_normalisasi_dict[term] \
        if term in kata_normalisasi_dict \
            else term for term in document]
def pp_stemming(document):
    stem_list = []
    for word in document:
        word_stemmed = stemmer.stem(word)
        if word_stemmed in remove_lists:
            continue
        stem_list.append(word_stemmed)
    return stem_list
def pp_stopword(document):
    list_clean = []
    for word in document:
        if word in sw:
            continue
        list_clean.append(word)
    return list_clean
def pp_text(text):
    pp = punc_casefold_lower_token(text)
    pp = pp_spelling(pp)
    # pp = pp_stemming(pp)
    pp = pp_stopword(pp)
    return pp

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        text = [request.form.get("review")]
        text = text[0]
        input_text = text

        text_pp = pp_text(text)

        text_list = []
        text_list.append(" ".join(text_pp))
        tmp = []
        tmp.append(text_list)
        text_list = tmp

        text_features = vectorize(text_list[0], tfidf_a)
        prediction_a = model_a.predict(text_features)
        output_a = prediction_a[0]
        prediction_s = ""

        if output_a=="application":
            text_features = vectorize(text_list[0], tfidf_sa)
            prediction_s = model_sa.predict(text_features)
        elif output_a=="service":
            text_features = vectorize(text_list[0], tfidf_ss)
            prediction_s = model_ss.predict(text_features)
        output_s = prediction_s[0]
        return render_template('index.html', input_text='Input: {}'.format(input_text), output_text_aspect='Aspect: {}'.format(output_a), output_text_sentiment='Sentiment: {}'.format(output_s))
    except:
        return render_template('index.html', input_text='Terjadi kesalahan')

if __name__ == "__main__":
    app.run(debug=True)