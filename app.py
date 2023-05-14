from flask import Flask, request, render_template
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')
russian_stop_words = stopwords.words("russian")
snowball = SnowballStemmer(language="russian")
def tokenize_sentence(text):
    text = text.replace(r'\n', ' ')
    pattern = "[^A-Za-zА-Яа-яЁё0-9]"
    text = re.sub(pattern," ",text)
    text = text.lower()
    tokens = word_tokenize(text, language="russian")
    tokens = [i for i in tokens if i not in string.punctuation]
    tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]
    return tokens


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    prediction = model.predict(request.form.values())  # features Must be in the form [[a, b]]

    output = prediction[0]
    if output == 1:
        output = 'Хороший'
    elif output == 2:
        output = 'Нейтральный'
    elif output == 3:
        output = 'Плохой'

    return render_template('index.html', prediction_text='Вид комментария:  {}'.format(output))
if __name__ == "__main__":
    app.run()