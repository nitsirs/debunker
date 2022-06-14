from flask import Flask, request, url_for, redirect, render_template, jsonify
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import tensorflow_hub as hub
import tensorflow_text
import json


app = Flask(__name__)
loaded_model = joblib.load("model")
tfidf = joblib.load("tfidf")
imported = tf.saved_model.load('TrainUSE')
use_model =imported.v.numpy()

module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3' 
model = hub.load(module_url)

def embed_text(input):
  return model(input)

df_news = pd.read_csv('df_news.csv')

def SearchDocument(query):
    q =query
    Q_Train =embed_text(q)
    
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
        a.loc[i,'img_src'] = df_news['img_src'][index]
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
    search_result = search_result[search_result['Score']>=0.4]
    search_result = search_result.replace(np.nan, '', regex=True)
    # res = search_result.iloc[0,:].headline + '\n' + search_result.iloc[0,:].tag + '\n' + search_result.iloc[0,:].url
    cards = []
    for i in range(search_result.shape[0]):
        card = {
            "thumbnailImageUrl": search_result.iloc[i,:].img_src,
            "imageBackgroundColor": "#FFFFFF",
            "title": search_result.iloc[i,:].headline,
            "text": search_result.iloc[i,:].tag,
            "defaultAction": {
                "type": "uri",
                "label": "ดูเพิ่มเติม",
                "uri": search_result.iloc[i,:].url,
            }
        }
        cards.append(card)
    response = {
        "fulfillmentMessages": [
      {
        "payload": {
            "line": {
              "type": "template",
              "altText": "this is a carousel template",
              "template": {
                "type": "carousel",
                "columns": cards,
                "imageAspectRatio": "rectangle",
                "imageSize": "cover"
              }
            }
          }
      }
    ],
        #"fulfillmentText": res,
        "source": "webhook"
    }
    if(search_result.shape[0] == 0):   
        encoded = tfidf.transform([text])
        prediction = loaded_model.predict(encoded)[0]
        proba = loaded_model.predict_proba(encoded).tolist()
        if(abs(proba[0][0]-proba[0][1]) <= 0.2):
            #response['fulfillmentText'] = "บอตไม่แน่ใจครับ เดี๋ยวรอแอดมินมาตอบนะครับ"
            response["fulfillmentMessages"] = [{"text": {"text": ["บอตไม่แน่ใจครับ เดี๋ยวรอแอดมินมาตอบนะครับ"]}}]
        elif(prediction == 0):
            #response['fulfillmentText'] = "บอตว่าอันนี้น่าจะเป็นข่าวจริงครับ อย่างไรก็ตาม ตรวจสอบข้อมูลก่อนแชร์ทุกครั้งนะครับ"
            response["fulfillmentMessages"] = [{"text": {"text": ["บอตว่าอันนี้น่าจะเป็นข่าวจริงครับ อย่างไรก็ตาม ตรวจสอบข้อมูลก่อนแชร์ทุกครั้งนะครับ"]}}]
        elif(prediction ==1):
            #response['fulfillmentText'] = "บอตว่าอันนี้น่าจะเป็นข่าวปลอมครับ คอยเฝ้าระวัง ตรวจสอบข้อมูลเพิ่มเติมก่อนแชร์นะครับ"
            response["fulfillmentMessages"] = [{"text": {"text": ["บอตว่าอันนี้น่าจะเป็นข่าวปลอมครับ คอยเฝ้าระวัง ตรวจสอบข้อมูลเพิ่มเติมก่อนแชร์นะครับ"]}}]
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
