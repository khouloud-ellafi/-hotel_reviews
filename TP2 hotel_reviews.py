# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:03:54 2020

@author: EBM
"""
import pandas as pd
import numpy as np
import seaborn as sns
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
#import spacy
import nltk


df = pd.read_csv('tripadvisor_hotel_reviews.csv')
print(' print the basic info of the data')
print(df.info())
print(df.shape)


df.Rating.value_counts()
Rating_count=df.groupby('Rating').count()
plt.bar(Rating_count.index.values, Rating_count['Review'])
plt.xlabel('Rating')
plt.ylabel('Number of Review')
plt.show()



df.head()

conditions = [
    (df['Rating'] > 3),
    (df['Rating'] < 3),
    (df['Rating'] == 3)
    ]

values = ['Positive','Negative','Neutral']

df['Sentiment'] = np.select(conditions, values)


count_positive, count_negative, count_neutral = df['Sentiment'].value_counts()
df_positive = df[ df['Sentiment'] == 'Positive']
df_negative =  df[ df['Sentiment'] == 'Negative']
df_neutral =  df[ df['Sentiment'] == 'Neutral']

print(count_positive)
print(count_negative)
print(count_neutral)
def wordcloud_draw(df, color = 'black'):
    words = ' '.join(df)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT' 
                                        ])
                                
    wordcloud = WordCloud(stopwords=STOPWORDS,background_color=color,width=2500,height=2000).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

print("Positive words")
wordcloud_draw(df_positive,'white')
print("Negative words")
wordcloud_draw(df_negative)

nltk.download('stopwords')
df.head()
df.shape
df.isnull().sum()
df["Rating"].value_counts()
df.loc[df["Review"] == ""]

pos = [5]
neg = [1, 2]
neu = [3, 4]

def sentiment(rating):
  if rating in pos:
    return 2
  elif rating in neg:
    return 0
  else:
    return 1  
df['Sentiment'] = df['Rating'].apply(sentiment)
df.head()

fig = go.Figure([go.Bar(x=df.Sentiment.value_counts().index, y=df.Sentiment.value_counts().tolist())])
fig.update_layout(
    title="Values in each Sentiment",
    xaxis_title="Sentiment",
    yaxis_title="Values")
fig.show()

from nltk.corpus import stopwords
stopwords_list = set(stopwords.words("english"))
punctuations = """!()-![]{};:,+'"\,<>./?@#$%^&*_~Â""" #List of punctuation to remove


def reviewParse(review):
    splitReview = review.split() 
    parsedReview = " ".join([word.translate(str.maketrans('', '', punctuations)) + " " for word in splitReview]) #Takes the stubborn punctuation out
    return parsedReview 
  
def clean_review(review):
    clean_words = []
    splitReview = review.split()
    for w in splitReview:
        if w.isalpha() and w not in stopwords_list:
            clean_words.append(w.lower())
    clean_review = " ".join(clean_words)
    return clean_review

df["Review"] = df["Review"].apply(reviewParse).apply(clean_review) #Parse all the reviews for their punctuation and add it into a new column

df.head()

df.head()
docs = list(df['Review'])[:7000]
from sklearn.feature_extraction.text import TfidfVectorizer 
 
tfidf_vectorizer=TfidfVectorizer(use_idf=True, max_features = 20000) 
 
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(docs)
X = tfidf_vectorizer_vectors.toarray()
Y = df['Sentiment'][:7000]

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV 
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score,roc_curve,auc
from sklearn.tree import DecisionTreeClassifier

SEED=123

X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)

fig = go.Figure([go.Bar(x=Y.value_counts().index, y=Y.value_counts().tolist())])
fig.update_layout(
    title="Values in each Sentiment",
    xaxis_title="Sentiment",
    yaxis_title="Values")
fig.show()

dt = DecisionTreeClassifier(random_state=SEED)
dt.fit(X_train,y_train)
y_pred_test = dt.predict(X_test)
print("Training Accuracy score: "+str(round(accuracy_score(y_train,dt.predict(X_train)),4)))
print("Testing Accuracy score: "+str(round(accuracy_score(y_test,dt.predict(X_test)),4)))

print(classification_report(y_test, y_pred_test, target_names=['positive', 'neutral', 'negative']))

cm = confusion_matrix(y_test, y_pred_test)
#print('Confusion matrix\n', cm)
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Negative', 'Actual Neutral', 'Actual Positive'], 
                        index=['Predict Negative', 'Predict Neutral', 'Predict Positive'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()
