from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__)

modelb = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('count_vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['text']
        data = cv.transform([user_input])
        prediction = modelb.predict(data)
        result = 'Stressed' if prediction[0] == 'Stress' else 'No Stress'
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
