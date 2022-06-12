from flask import Flask, request, url_for, redirect, render_template, jsonify
import joblib

app = Flask(__name__)
loaded_model = joblib.load("model")
tfidf = joblib.load("tfidf")

@app.route('/')
def index():
    return "Debunker is a machine learning API that classifies news articles as true or false. It is designed to help journalists and researchers identify and debunk fake news stories. Debunker uses machine learning algorithm to analyze article content. The API returns a score indicating the likelihood that a story is fake. Debunker is free and open source, and can be used online."


@app.route('/webhook', methods=['POST'])
def webhook():
    loaded_model = joblib.load("model")
    tfidf = joblib.load("tfidf")   
    data = request.get_json(force=True)
    text = data['queryResult']['queryText']
    encoded = tfidf.transform(text)
    prediction = loaded_model.predict(encoded)[0]
    proba = loaded_model.predict_proba(encoded).tolist()
    response = {
        "fulfillmentText": "บอตน้อยไม่แน่ใจครับ เดี๋ยวรอแอดมินมาตอบนะครับ",
        "source": "webhook"
    }
    #if(prediction == 0):
    #    response["fulfillmentText"] = "อันนี้เป็นข่าวจริงครับ"
    #else:
    #    response["fulfillmentText"] = "อันนี้ข่าวปลอมครับ"
    return({
        "fulfillmentText": "บอตน้อยไม่แน่ใจครับ เดี๋ยวรอแอดมินมาตอบนะครับ",
        "source": "webhook"
    })


@app.route('/classifier_api', methods=['POST'])
def classifier_api():
    req = request.get_json(force=True)
    text = [req['text']]
    encoded = tfidf.transform(text)
    prediction = loaded_model.predict(encoded)[0]
    proba = loaded_model.predict_proba(encoded).tolist()
    return jsonify(result=int(prediction), proba=proba)


if __name__ == '__main__':
    app.run(debug=True)
