#AI Phase wise project submission

#Building a smarter AI-powerd spam classification

Data source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

# DESIGN PROCESS:
There are many different ways to design a spam filtering system, but one common approach is to use machine learning algorithms. These algorithms can be trained on data sets of known and non-spam emails and then used to classify new emails.

```bash
   git clone https://github.com/Chandru53/aiphase1.git
   ```

# innovation steps:
  1.Load and simplify the dataset. ...
2.Explore the dataset: Bar Chart. ...
3.Explore the dataset: Word Clouds. ...
4.Handle imbalanced datasets. ...
5.Split the dataset. ...
6.Apply Tf-IDF Vectorizer for feature extraction. ...
7.Train our Naive Bayes Model. ...
8.Check out the accuracy, and f-measure.

# import the required packages:
      %matplotlib inline
import matplotlib.pyplot as plt
import csv
import sklearn
import pickle
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import

# code:
data = pd.read_csv('dataset/spam.csv', encoding='latin-1')
data.head()
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v2" : "text", "v1":"label"})
data[1990:2000]

# Import nltk packages and Punkt Tokenizer Models
import nltk
nltk.download("punkt")
import warnings
warnings.filterwarnings('ignore')

# Creating a corpus of spam messages
for val in data[data['label'] == 'spam'].text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        spam_words = spam_words + words + ' '

# Creating a corpus of ham messages
for val in data[data['label'] == 'ham'].text:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        ham_words = ham_words + words + ' '

#Spam Word cloud
plt.figure( figsize=(10,8), facecolor='w')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

#Creating Ham wordcloud
plt.figure( figsize=(10,8), facecolor='g')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# classifiers used:
1.spam classifier using logistic regression
2.email spam classification using 3.Support Vector Machine(SVM)
4.spam classifier using naive bayes
5.spam classifier using decision tree
6.spam classifier using K-Nearest Neighbor(KNN)
7.spam classifier using Random Forest Classifier.
