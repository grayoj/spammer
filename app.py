# Import String
import string
from sklearn.model_selection import train_test_split

# Import Streamlit.
import streamlit as st


# Import pandas to read CSV files.
import pandas as pd

# Import natural language toolkit.
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Import Sickitlearn
import sklearn

# Import Naive Bayes Module
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
# Import module to display accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('dataset/data.csv')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis =1)
df.rename(columns = {'v1':'labels', 'v2': 'message'}, inplace = True)
df.drop_duplicates(inplace = True)
df['labels'] = df['labels'].map({'ham': 0, 'spam': 1})
print(df.head())

def clean_data(message):
    message_without_labels = message
    message_without_punc = [character for character in message if character not in string]
    message_without_punc = ''.join(message_without_punc)
    
    separator = ''
    return separator.join([word for word in message_without_punc.split() if word.lower() not in stopwords.words('english')])

df['message'] = df['message'].apply(clean_data)
x = df['message']
y = df['labels']

cv = CountVectorizer()

x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

model = MultinomialNB().fit(x_train, y_train)
predictions = model.predict(x_test)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

def predict(text):
    labels = ['This is not a Spam', 'This is Spam']
    x = cv.fit_transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int(''.join(s))
    return str('This message looks like a spam message.')


