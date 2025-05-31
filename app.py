import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('best_rf.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vector.pkl', 'rb'))

# Load dataset (Modify the path accordingly)
news_df = pd.read_csv(r"C:\Users\Aasher\cleaned_news_dataset.csv")  # Ensure this matches your dataset file

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_news')
def get_news():
    # Drop rows where title or text are NaN
    clean_news_df = news_df.dropna(subset=['title', 'text'])  
    
    # Optionally, replace NaN values with a default text instead
    clean_news_df['text'] = clean_news_df['text'].fillna("No content available")
    
    news_data = clean_news_df[['title', 'text']].to_dict(orient='records')
    
    print("Sending News Data:", news_data)  # Debugging step
    return jsonify({"articles": news_data})

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    input_data = vectorizer.transform([news_text])
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    print("Prediction:", prediction)
    print("Probabilities:", probabilities)

    # Check which class = Fake
    result = "Fake News" if prediction[0] == 1 else "Real News"
    confidence = round(float(probabilities[0][prediction[0]]) * 100, 2)
    return render_template('index.html', prediction=f"{result} (Confidence: {confidence}%)")


if __name__ == '__main__':
    app.run(debug=True)