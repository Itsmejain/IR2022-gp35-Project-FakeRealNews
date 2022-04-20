# from flask import Flask
from flask import Flask, flash,jsonify, redirect, render_template, request, url_for
from random import randrange
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
# from werkzeug.datastructures import  FileStorage
import pandas as pd
# import os
import pytesseract
# import shutil
import os
from PIL import Image
import random
# from os.path import join, dirname, realpath
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from wordcloud import STOPWORDS
import nltk
import pickle
nltk.download('punkt')
import re
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.tokenize import word_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS




from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
# from sklearn import metrics
# from sklearn.metrics import classification_report


# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# # from wordcloud import WordCloud, STOPWORDS
# import nltk
# nltk.download('punkt')
# import re
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.corpus import stopwords
# nltk.download("stopwords")
# from nltk.tokenize import word_tokenize, sent_tokenize
# import gensim
# from gensim.utils import simple_preprocess
# from gensim.parsing.preprocessing import STOPWORDS
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
# from sklearn import metrics
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# import pickle
# from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
# from tensorflow.keras.models import Model

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = { 'jpg', 'jpeg'}

app = Flask(__name__, 
            template_folder="./templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


#route home api
@app.route("/")
def home():
 return render_template('index.html')# return "<p>Hello, World!</p>"


# //THIS API WILL FETCH THE RANDOM DATA FROM THE DATASET AND DISPLAY ON UI
@app.route('/r',methods=['GET'])
def randomnews():
    data = pd.read_csv("dataset/combineddataset.csv")
    index = randrange(0, len(data)-1, 1)
    title = data.loc[index].title
    text = data.loc[index].text
    # return "<p>helllo {text}</p>"
    # return jsonify({'title': title, 'text': text})
    return render_template('index.html', text = text)


@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return 'NO FILE CHOSEN'
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            # flash('No selected file')
            return 'NO FILE CHOSEN'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],"testing.jpg" ))
            return image_processing()#render_template('/r')
        else:
            return 'Incorrect File Format'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           

@app.route('/imageprocessing', methods=['GET', 'POST'])
def image_processing():
    pytesseract.pytesseract.tesseract_cmd = "/app/.apt/usr/bin/tesseract" 
    #Use above when on server
    # pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    IMAGEPATH = "uploads/"
    img = "testing.jpg"
    extractedInformation = pytesseract.image_to_string(Image.open(IMAGEPATH+img))
    # print(extractedInformation)
    return render_template('index.html', text = extractedInformation)




try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    text = pytesseract.image_to_string(Image.open(filename))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text

## CLASSIFICATION MODEL

@app.route('/classification', methods=['GET', 'POST'])
def news_classification():
    if request.method == 'POST':
      result = request.data.decode('UTF-8')
      print(result)
      #Here we are now getting the value of the input from flask UI

      title = ""
      text = result
      a=[]
      b=[]
      a.append(title)
      b.append(text)
      data=pd.DataFrame({'title':a,'text':b})
      list_=preprocess_RFC(data) 
      print(data)
      print(list_)
      path='RFC'
      with open(path , 'rb') as f:
         lr = pickle.load(f)
      print('successs')
      p=lr.predict(list_)
      print(p[0])
      #TRY WITH DUMMY FAKE DATA - tested working fine
      if p[0]==0:
          prediction_fr = "FAKE"
          print(prediction_fr)
      else:
          prediction_fr = "REAL"
          print(prediction_fr)
    example_embed='This string is from python'
    # return render_template('index.html',pred =prediction_fr)
    return prediction_fr


def remove_stopWords(txt):
  stopWords = stopwords.words('english')
  stopWords.extend(['from', 'subject', 're', 'edu', 'use'])
  result = []
  for token in gensim.utils.simple_preprocess(txt):
      if token not in gensim.parsing.preprocessing.STOPWORDS:
        if len(token) > 3:
           if  token not in stopWords:
             result.append(token)
  return result

def get_list(txt):
  list_words = []
  for i in txt:
      for j in i:
          list_words.append(j)
  new_list = len(list(set(list_words)))
  return new_list
  
def preprocess_RFC(data):
  data['title_text_merged'] = data['title'] + ' ' + data['text']
  data['tokenization'] = data['title_text_merged'].apply(remove_stopWords)
  new_list=get_list(data.tokenization)
  data['tokenization_merged'] = data['tokenization'].apply(lambda x: " ".join(x))   
  return data.tokenization_merged



@app.route('/recommendation', methods=['GET', 'POST'])
def news_recommendation():
    if request.method == 'POST':
      result = request.data.decode('UTF-8')
    #   print(result)
    df = pd.read_csv("dataset/True.csv") 
    features = ['text', 'subject']
    for feature in features:
      df[feature] = df[feature].fillna('')
    df["combined_features"] = df.apply(combined_features, axis =1)
    df1 = pd.DataFrame(df["combined_features"])
    v = result
    df1=df1.append({"combined_features":v},ignore_index=True)
    print('Running count Vectorizer')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df1["combined_features"])
    print("Count Matrix:", count_matrix.toarray())
    print('Running Cosine Similarity .....')
    cosine_sim = cosine_similarity(count_matrix)
    recommended_list = list(enumerate(cosine_sim[-1]))
    sorted_recommended_list = sorted(recommended_list, key=lambda x:x[1], reverse=True)
    sorted_recommended_list=sorted_recommended_list[1:]
    i=0
    newslist = []
    for news in sorted_recommended_list:
     print('news list printing ')
     i=i+1
     if i>11:
        break
     newslist.append(str(i)+'.'+df['title'][news[0]]+":\n"+(df['text'][news[0]])[:200]+"...."+"\n")
        # print(get_title_from_indexee(movie[0])+":\n"+(df['text'][movie[0]])[:500]+"....")
        # print()
     print(df['title'][news[0]])
    #  print(get_title_from_indexee(movie[0]))
    return render_template('index.html', newslist=newslist , text = result) 
    # return newslist

def get_title_from_indexee(index):
    return df['title'][index]
def combined_features(row):
    return row['title']+" "+row['text']+" "+row['subject']


# print(ocr_core('images/ocr_example_1.png'))

#for local server
if __name__ == "__main__":
 app.run(debug=True,port=8000)

# import logging

# app.logger.addHandler(logging.StreamHandler(sys.stdout))
# app.logger.setLevel(logging.ERROR)
