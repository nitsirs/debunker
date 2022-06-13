from flask import Flask, request, url_for, redirect, render_template, jsonify
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import spacy_universal_sentence_encoder
import json


app = Flask(__name__)
loaded_model = joblib.load("model")
tfidf = joblib.load("tfidf")
imported = tf.saved_model.load('TrainUSE')
use_model =imported.v.numpy()
nlp = spacy_universal_sentence_encoder.load_model('xx_use_md')

df_news = pd.read_csv('df_news.csv')

def SearchDocument(query):
    q =query
    Q_Train =[nlp(q).vector]
    
    linear_similarities = linear_kernel(Q_Train, use_model).flatten() 
    Top_index_doc = linear_similarities.argsort()[:-11:-1]

    linear_similarities.sort()
    a = pd.DataFrame()
    for i,index in enumerate(Top_index_doc):
        a.loc[i,'index'] = str(index)
        a.loc[i,'headline'] = df_news['headline'][index] ## Read File name with index from File_data DF
        try:
          a.loc[i,'tag'] = df_news['tag'][index]
        except:
          pass
        a.loc[i,'url'] = df_news['url'][index]
    for j,simScore in enumerate(linear_similarities[:-11:-1]):
        a.loc[j,'Score'] = simScore
    return a


@app.route('/')
def index():
    return "Debunker is a machine learning API that classifies news articles as true or false. It is designed to help journalists and researchers identify and debunk fake news stories. Debunker uses machine learning algorithm to analyze article content. The API returns a score indicating the likelihood that a story is fake. Debunker is free and open source, and can be used online."


@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(force=True)
    text = data['queryResult']['queryText']
    search_result = SearchDocument(text)
    search_result = search_result.to_json(orient="split")
    parsed = json.loads(search_result)
    parsed = json.dumps(parsed, indent=4)

    encoded = tfidf.transform([text])
    prediction = loaded_model.predict(encoded)[0]
    proba = loaded_model.predict_proba(encoded).tolist()
    response = {
        "search_result" : parsed,
        "source": "webhook"
    }
    print(proba)
    if(abs(proba[0][0]-proba[0][1]) <= 0.2):
        response["fulfillmentText"] = "บอตน้อยไม่แน่ใจครับ เดี๋ยวรอแอดมินมาตอบนะครับ"
    elif(prediction == 0):
        response["fulfillmentText"] = "อันนี้เป็นข่าวจริงครับ"
    elif(prediction ==1):
        response["fulfillmentText"] = "อันนี้ข่าวปลอมครับ"
    return(response)


@app.route('/classifier_api', methods=['POST'])
def classifier_api():
    req = request.get_json(force=True)
    text = req['text']
    encoded = tfidf.transform(text)
    prediction = loaded_model.predict(encoded)[0]
    proba = loaded_model.predict_proba(encoded).tolist()
    return jsonify(result=int(prediction), proba=proba)


if __name__ == '__main__':
    app.run(debug=True)
