from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from joblib import load

app = Flask(__name__)

# Load the trained model and vectorizer
model = load('model.joblib')
vectorizer = load('vectorizer.joblib')  # Assuming you saved the vectorizer during training

def predict_genre(description):
    # Transform the input description using the loaded vectorizer
    new_text_features = vectorizer.transform([description])
    
    # Predict the genre using the trained model
    predicted_genre = model.predict(new_text_features)[0]
    
    return predicted_genre

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        description = request.form['description']
        
        # Predict genre for the input description
        predicted_genre = predict_genre(description)
        
        return render_template('index.html', description=description, predicted_genre=predicted_genre)

if __name__ == '__main__':
    app.run(debug=True)
