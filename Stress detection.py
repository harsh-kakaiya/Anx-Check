import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("dataset\\stress.csv")
print(data.head())

#checking null values
print(data.isnull().sum())

#no null values
#cleaning
import nltk
import re
nltk.download('stopwords')
stemmer=nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text
data["text"] = data["text"].apply(clean)

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ".join(i for i in data.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords = stopword, background_color = "white").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#model
data["label"]=data["label"].map({0:"No Stress", 1: "Stress"})
data = data[["text", "label"]]
print(data.head())

#split dataset into train and test
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

x = np.array(data["text"])
y = np.array(data["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(xtrain, ytrain)

# import joblib
# joblib.dump(model, 'stress_model.joblib')

pickle.dump(model, open('model.pkl' , 'wb'))
modelb = pickle.load(open('model.pkl' , 'rb'))

# Save the CountVectorizer
pickle.dump(cv, open('count_vectorizer.pkl', 'wb'))


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)

user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)

from sklearn.metrics import accuracy_score

y_pred = model.predict(xtest)
score = accuracy_score(ytest, y_pred)
print(f'Accuracy: {score}')

model.score(xtrain,ytrain)



