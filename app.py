from flask import Flask,request, url_for, redirect, render_template, jsonify
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classifier_api',methods=['POST'])
def classifier_api():
    req = request.get_json(force=True)
    text = [req['text']]
    loaded_model = joblib.load("model")
    tfidf = joblib.load("tfidf")
    encoded = tfidf.transform(text)
    prediction = loaded_model.predict(encoded)[0]
    proba = loaded_model.predict_proba(encoded).tolist()
    return jsonify(result = int(prediction), proba=proba)

if __name__ == '__main__':
    app.run(debug=True)