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

#Create an app object using the Flask class.
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('model.pkl', 'rb'))

#Define the route to be home.
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory.
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder.

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission.
#Redirect to /predict page with the output
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


#When the Python interpreter reads a source file, it first defines a few special variables.
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__).
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here.
#If we import this file (module) to another file then __name__ == app (which is the name of this python file).

if __name__ == "__main__":
    app.run()