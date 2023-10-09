import numpy as np
import matplotlib.pyplot as plt
import nltk
import tensorflow as tf
import keras 
import pandas as pd
import string
import nlp_utils as nu
f = open("dialogs.txt","r")
print(f.read())
df= pd.read_csv('dialogs.txt',names=('Query','Response'),sep=('\t'))
df
df.shape
df.columns
df.info()
df.describe()
df.nunique()
df.isnull().sum()
df['Query'].value_counts()
df['Response'].value_counts()

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
Text = df['Query']
sia = SentimentIntensityAnalyzer()
for sentence in Text:
    print(sentence) 
    
    ss = sia.polarity_scores(sentence)
    for k in ss:
        print('{0}:{1}, ' .format(k, ss[k]), end='')
    print()

analyzer = SentimentIntensityAnalyzer()
df['rating'] = Text.apply(analyzer.polarity_scores)
df=pd.concat([df.drop(['rating'], axis=1), df['rating'].apply(pd.Series)], axis=1)
df

from wordcloud import WordCloud
def wordcloud(df, label):
     
    subset=df[df[label]==1]
    text=df.Query.values
    wc=WordCloud(background_color="black", max_words=1000)

    wc.generate(" ".join(text))

    plt.figure(figsize=(20, 20))
    plt.subplot(221)
    plt.axis("off")
    plt.title("Words frequented in  {}".format(label), fontsize=20)
    plt.imshow(wc.recolor(colormap='gist_earth', random_state=244), alpha=0.98)

wordcloud(df, 'Query')
wordcloud(df,'Response')

import re
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation),' ', x.lower())
remove_n = lambda x: re.sub("\n", " ", x)
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]', r' ', x)
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
df['Query'] = df['Query'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)
df['Response'] = df['Response'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)
df      #final cleaned dataset

pd.set_option('display.max_rows', 3000)
df

imp_sent = df.sort_values(by='compound', ascending=False)
imp_sent.head(5)
pos_sent = df.sort_values(by='pos', ascending=False)
pos_sent.head(5)
neg_sent = df.sort_values(by='neg', ascending=False)
neg_sent.head(5)
neu_sent = df.sort_values(by='neu', ascending=False)
neu_sent.head(5)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
factors = tfidf.fit_transform(df['Query']).toarray()
tfidf.get_feature_names_out()

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
def lemmatization_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    words = sentence.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words) 

from sklearn.metrics.pairwise import cosine_distances
query ='who are you ?'
def chatbot(query):
    query = lemmatization_sentence(query)
    query_vector = tfidf.transform([query]).toarray()
    similar_score = 1-cosine_distances(factors, query_vector)
    index = similar_score.argmax()
    matching_questions = df.loc[index]['Query']
    response = df.loc[index]['Response']
    pos_score = df.loc[index]['pos']
    neg_score = df.loc[index]['neg']
    neu_score = df.loc[index]['neu']
    confidence = similar_score[index][0]
    chat_dict = {'match' : matching_questions,
                 'response': response,
                 'score': confidence,
                 'pos': pos_score,
                 'neg': neg_score,
                 'neu': neu_score}
    return chat_dict
    return response

while True:
    query = input('USER: ')
    if query == 'exit':
        break
    response = chatbot(query)
    if response['score'] <= 0.2:
        print('BOT: Please rephrase your Question,')
    else:
        print('='*80)
        print('logs:\n Matched Questions: %r\n Confidence Score: %0.2f \n PositiveScore: %r\n NegativeScore: %r \n NeutralScore: %r'%(response['match'], response['score']*100, response['pos'],response['neg'],response['neu'])) 
        print('='*80)
    print('BOT: ', response['response'])       


       