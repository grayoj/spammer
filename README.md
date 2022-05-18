# Streamlit Spam Classification

A quite simple application that demonstrates the use of Naive Bayes algorithm for multinomial models, pandas for reading datasets and other sickitlearn models.

## Requirements

- Python 3
- Pip (To Install packages)
- Streamlit 
- Pandas
- Numpy
- Sickitlearn

## Installation

``Step 1``

Clone this repository. 
```shell
git clone https://github.com/grayoj/spam-detection.git
```

Install the <a href="http://python.org">Python Programming Language</a>

``Step 2``
Using Pip, Install <a href="http://streamlit.com">Streamlit</a> which is the server for the application.
```shell
$ pip install streamlit
```

``Step 3``
Install pandas to read data sets
```shell
$ pip install pandas
```

``Step 4``
Install Sickitlearn modules that contains all necessary modules used in the project
```shell
$ pip install sklearn
```

``Step 5``
Install Numpy for mathematical functions.
```shell
$ pip install numpy
```

``Step 6``
Install the Natural language model toolkit.
```shell
$ pip install nltk
```

If you use Pylance, it should validate all imports made.

```python
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
```

The imports as above.

## Run Application

```shell
cd spam-detection
streamlit run app.py
```

Done. Should be running on the localhost.