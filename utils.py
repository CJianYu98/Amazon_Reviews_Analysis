# General packages
import os 
import datetime as dt
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import json
import glob
import pickle
# from clf import *

# Text Processing
import nltk
from nltk import word_tokenize, TweetTokenizer, sent_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import gensim
from gensim.models import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Topic Modelling
import gensim
from gensim.models import CoherenceModel
import pyLDAvis.gensim
from corextopic import corextopic as ct
import corextopic.vis_topic as vt

# Plots
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


# EDA
def plot_num_reviews_per_category(file_path):

    csv_files = [filename for filename in os.listdir(file_path) if filename.endswith('reviews.csv')]

    num_reviews = {
        'product': [],
        'reviews': []
    }

    for filename in csv_files:
        df = pd.read_csv(f'{file_path}/{filename}')
        product_name = filename[:-4] # remove '.csv' at the end of each csv file
        num_reviews['product'].append(product_name)
        num_reviews['reviews'].append(len(df))

    # matplotlib.use('TKAgg')
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,2])
    product = num_reviews['product']
    reviews_num = num_reviews['reviews']
    ax.bar(product, reviews_num)
    plt.title("Bar chart of number of reviews for each product category", fontsize = 20)
    plt.xlabel("Products", fontsize = 15)
    plt.ylabel("Number of reviews",fontsize = 15)
    for i in range(len(num_reviews['product'])):
        plt.text(x=i-0.20, y = num_reviews['reviews'][i]+5000 , s=f"{num_reviews['reviews'][i]}" , fontdict=dict(fontsize=15))
    # ax.set_xticks()
    # plt.tight_layout()
    plt.show()

def drop_duplicate_spam(df):
    """
    Drop duplicates or spam reviews

    Args:
        df ([pd.DataFrame]): DataFrame which contains duplicates or spam reviews

    Returns:
        [pd.DataFrame]: DataFrame with duplicates or spam reviews removed
    """

    # Duplicates
    df.drop_duplicates(subset=['Author', 'ReviewID', 'Date', 'Content', 'ProductID'], inplace=True)

    # User leaving more than 1 review for a product in a day
    df.drop_duplicates(subset=['Author', 'Date', 'ProductID'], inplace=True)

    # Multiple reviews with same review message for a product in the same day (possibly using multiple accounts)
    df.drop_duplicates(subset=['Date', 'Content', 'ProductID'], inplace=True)
    
    return df




# Sentiments
def sent_tokenize_to_df(df):
    sentences_dict = {'Content': [], 'Product_Name': [], 'ProductID': []}

    for i, row in df.iterrows():
        sentences = sent_tokenize(df['Content'].iloc[i])
        for sent in sentences:
            sentences_dict['Content'].append(sent)
            sentences_dict['Product_Name'].append(df['Product_Name'].iloc[i])
            sentences_dict['ProductID'].append(df['ProductID'].iloc[i])

    final_df = pd.DataFrame(sentences_dict)
    # final_df.drop_duplicates(subset=['Sentences'], inplace=True)
    final_df.reset_index(inplace=True)
    return final_df

def vader_compound_score(x):
    vader_analyser = SentimentIntensityAnalyzer()
    score = vader_analyser.polarity_scores(x)
    return score['compound']

def vader_sentiment(df):
    df['Vader_compound_score'] = df['Content'].apply(lambda x: vader_compound_score(x))
    return df

def textblob_sentiment(df):
    df['tb_polarity'] = df['Content'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['tb_subjectivity'] = df['Content'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df

def emotions(df):
    emotions_clf = pickle.load(open('tfidf_svm.sav', 'rb'))
    df['emotion'] = df['Sentences'].apply(lambda x: emotions_clf.predict([x]))
    return df

def get_sentiments(df):
    df = vader_sentiment(df)
    df = textblob_sentiment(df)
    df['final_sentiment'] = df['Vader_compound_score'] + df['tb_polarity']

    return df

def print_top_reviews(df, pid, reviews, n):
    print('Product ID:\n' + '   - ' + pid + '\n')

    names = list(set(df['Product_Name'].to_list()))
    for i in range(len(names)):
        if i == 0:
            print(f'Product Name:\n   - {names[i]}')
        else:
            print(f'  - {names[i]}')
    print()
    print('-'*100)
    print()
        
    for i in range(n):
        print(f'Rank {i+1} review:')
        print(f'{reviews[i]}')
        print()

def get_top_positive_reviews(category, pid, n):
    df = pd.read_csv(f'./Data/Cleaned Data/{category}_reviews.csv', index_col=0)
    df = df[df['ProductID'] == pid].reset_index()
    sent_df = sent_tokenize_to_df(df)

    df_sentiments = get_sentiments(sent_df)

    reviews = df_sentiments.sort_values(by='final_sentiment', ascending=False)['Content'].to_list()

    print_top_reviews(df_sentiments, pid, reviews, n)

def get_top_negative_reviews(category, pid, n):
    df = pd.read_csv(f'./Data/Cleaned Data/{category}_reviews.csv', index_col=0)
    df = df[df['ProductID'] == pid].reset_index()
    sent_df = sent_tokenize_to_df(df)

    df_sentiments = get_sentiments(sent_df)

    reviews = df_sentiments.sort_values(by='final_sentiment', ascending=True)['Content'].to_list()

    print_top_reviews(df_sentiments, pid, reviews, n)

def get_top_reviews_by_emotions(category, pid, emotion, n):
    df = pd.read_csv(f'./Data/Cleaned Data/{category}_reviews.csv', index_col=0)
    df = df[df['ProductID'] == pid].reset_index()
    sent_df = sent_tokenize_to_df(df)

    df_sentiments = get_sentiments(sent_df)

    reviews_emotion = df_sentiments[df_sentiments['emotion'] == emotion].reset_index()

    if emotion == 'joy':
        reviews = reviews_emotion.sort_values(by='final_sentiment', ascending=False)['Sentences'].to_list()
    else:
        reviews = reviews_emotion.sort_values(by='final_sentiment', ascending=True)['Sentences'].to_list()
    

    print_top_reviews(reviews_emotion, pid, reviews, n)

def get_corex_top_positive_reviews(topics_df, pid, topic, n):
    df = topics_df[topics_df[f'topic_{topic}'] == 1.0].reset_index()

    sent_df = sent_tokenize_to_df(df)

    df_sentiments = get_sentiments(sent_df)

    reviews = df_sentiments.sort_values(by='final_sentiment', ascending=False)['Content'].to_list()

    print_top_reviews(df_sentiments, pid, reviews, n)

def get_corex_top_negative_reviews(topics_df, pid, topic, n):
    df = topics_df[topics_df[f'topic_{topic}'] == 1.0].reset_index()

    sent_df = sent_tokenize_to_df(df)

    df_sentiments = get_sentiments(sent_df)

    reviews = df_sentiments.sort_values(by='final_sentiment', ascending=True)['Content'].to_list()

    print_top_reviews(df_sentiments, pid, reviews, n)





# LDA_Mallet 
def get_category_reviews(category):
    df = pd.read_csv(f'./Data/Cleaned Data/{category}_reviews.csv')
    sent_df = sent_tokenize_to_df(df)

    return sent_df

def get_product_reviews(category, product_id):
    df = pd.read_csv(f'./Data/Cleaned Data/laptops_reviews.csv')
    df = df[df['ProductID'] == product_id].reset_index()
    sent_df = sent_tokenize_to_df(df)

    return sent_df

def corpus2docs(df, stop_list):
    docs1 = [word_tokenize(comment) for comment in df['Content']]
    docs2 = [[w.lower() for w in doc] for doc in docs1]
    docs3 = [[w for w in doc if len(w) > 2] for doc in docs2]
    docs4 = [[w for w in doc if re.search('^[a-z]+$', w)] for doc in docs3]
    docs5 = [[w for w in doc if w not in stop_list] for doc in docs4]
    return docs5

def docs2vecs(docs, dic):
    vecs = [dic.doc2bow(doc) for doc in docs]
    return vecs

def get_coherence_scores(mallet_path, docs, dic, vecs):
    model_list = []
    coherence_values = []
    model_topics = []

    for num_topics in range(2, 15):
        lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path, iterations=500, corpus=vecs, num_topics=num_topics, id2word=dic, random_seed=99)
        coherencemodel = CoherenceModel(model=lda_mallet, texts=docs, dictionary=dic, coherence='c_v')
        model_topics.append(num_topics)
        model_list.append(lda_mallet)
        coherence_values.append(coherencemodel.get_coherence())
        print("#Topics: " + str(num_topics) + " Score: " + str(coherencemodel.get_coherence()))

    return coherence_values

def coherence_plot(limit, coherence_values):
    x = range(2, limit, 1)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

def get_topics(lda_mallet, no_topics):
    vecTop = lda_mallet.show_topics(num_words=20)
    for i in range(0, no_topics):
        print(vecTop[i])
        print()

def create_pyLDAvis(lda_mallet, vecs, dic, filename):
    mallet_lda_model= gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_mallet)

    pyLDAvis.enable_notebook()
    visual= pyLDAvis.gensim.prepare(mallet_lda_model, vecs, dic)
    pyLDAvis.save_html(visual, f"{filename}_viz.html")

def format_topics_sentences(ldamodel, corpus, data, ori_data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(data)
    ori_text = ori_data['Content']
    sent_topics_df = pd.concat([sent_topics_df, contents, ori_text], axis=1)
    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Tokenized Text', 'Content']

    return df_dominant_topic

def get_reviews_by_topic(df, topic_no):
    df = df[df['Dominant_Topic'] == topic_no]
    df = df.sort_values(by='Topic_Perc_Contrib', ascending=False)
    return df

def generate_wordcloud(df, polarity = None, stop_words = []):
    if polarity != None:
        df = get_sentiments(df)

        if polarity == 'pos':
            df = df[df['final_sentiment'] >= 0.01]
        elif polarity == 'neg':
            df = df[df['final_sentiment'] <= -0.01]
        else:
            df = df[(df['final_sentiment'] > -0.01) & (df['final_sentiment'] < 0.01)]
    
    text = " ".join(one_row for one_row in df['Content'])

    stop_list = stopwords.words('english')
    stop_list += stop_words

    # Create and generate a word cloud image:
    wordcloud = WordCloud(stopwords=stop_list, background_color="white", width=1000, height=500).generate(text)

    # Display the generated image:
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()




# CorEx Topic Modelling
def get_corex_vectorizer_vocab(df):
    comments = []
    for row in df['Content']:
        text_tokenize = word_tokenize(row)
        text_lower = [w.lower() for w in text_tokenize]
        text_words_only = [w for w in text_lower if re.search('^[a-z]+$',w)]
        text_drop_2letters = [w for w in text_words_only if len(w) > 2]
        text_joined = ' '.join(text_drop_2letters)
        comments.append(text_joined)

    df['Reviews'] = comments

    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    vectorizer = vectorizer.fit(df['Reviews'])
    vecs = vectorizer.transform(df['Reviews'])
    vocab = vectorizer.get_feature_names()

    return vecs, vocab

def print_corex_top_topic_words(model, n_words):
    top_words = {}

    for i, topic_ngrams in enumerate(model.get_topics(n_words=n_words)):
        topic_ngrams = [ngram[0] for ngram in topic_ngrams if ngram[1] > 0]
        top_words[f'topic{i}'] = topic_ngrams
        print("Topic #{}: {}".format(i+1, ", ".join(topic_ngrams)))
        print()
    
def plot_corex_total_correlation(model):
    plt.figure(figsize=(10,5))
    plt.bar(range(model.tcs.shape[0]), model.tcs, color='#4e79a7', width=0.5)
    plt.xlabel('Topic', fontsize=16)
    plt.ylabel('Total Correlation (nats)', fontsize=16)

def corex_label_topics(df, model, vecs, no_topics):
    topic_df = pd.DataFrame(model.transform(vecs), columns=["topic_{}".format(i+1) for i in range (no_topics)]).astype(float)
    topic_df.index = df.index
    df = pd.concat([df, topic_df], axis=1)
    return df




def label_subjectivity(x):
    if x >= 0.30:
        return 1
    return 0

def get_reviews_by_subjectivity(df, subjectivity):
    df['subjectivity'] = df['tb_subjectivity'].apply(lambda x: label_subjectivity(x))

    if subjectivity == 'opinion':
        out_df = df[df['subjectivity'] == 1]
    else:
        out_df = df[df['subjectivity'] == 0]
    
    return out_df
