import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from google.colab import drive
drive.mount('/content/drive')

#labels = ['England', 'Ireland', 'Scotland']

df = pd.read_csv('drive/MyDrive/train_data.csv')
df_test = pd.read_csv('drive/MyDrive/test_data.csv')

stopwords_all = list(stopwords.words('danish') + stopwords.words('german') + stopwords.words('spanish') + stopwords.words('italian') + stopwords.words('dutch'))
wnl = WordNetLemmatizer()

def proceseaza(text):
    text = text.replace('\n', ' ')
    text = text.strip()
    text = text.lower()
    text = ''.join(x for x in text if not x.isdigit())
    text = ' '.join(word for word in text.split() if word not in stopwords_all)
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    return text

df["text"] = df.text.map(proceseaza)
df.text = [wnl.lemmatize(word) for word in df.text]
df_test["text"] = df_test.text.map(proceseaza)
df_test.text = [wnl.lemmatize(word) for word in df_test.text]

texts = df.text
labels = df.label

#folosit initial pentru a calcula local acuratetea
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

model = Pipeline([('vect', CountVectorizer(min_df=3)),
                ('tfidf', TfidfTransformer(sublinear_tf=True)),
                ('clf', svm.LinearSVC(C=0.5, max_iter=5000)),
               ])
model.fit(texts, labels)

predictions = model.predict(df_test.text)

import numpy as np

testid = np.arange(1, 13861)

np.savetxt("drive/MyDrive/submisie_svm_new.csv", np.stack((testid, predictions)).T, fmt="%s", delimiter=",", header="id,label", comments='')