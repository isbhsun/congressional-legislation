import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from scipy import stats
import joblib
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import gensim
from nltk.tokenize import word_tokenize


bill_text_df = pd.read_csv('../results/116bill_text.csv')

vectorizer = CountVectorizer(max_df=0.9,
                             min_df=3, 
                             max_features=6000)

word_vec = vectorizer.fit_transform(bill_text_df['text_bigrams'])

search_params = {'n_components': [20, 30, 32, 35, 37, 40],
                 'learning_decay': [.5, .55, .6, .65, .7],
                 'doc_topic_prior': [0.025, 0.05, 0.075, 0.1],
                 'topic_word_prior': [0.025, 0.05, 0.075, 0.1],
                 'learning_method': ['online'],
                 'random_state': [0],
                 'n_jobs': [-1]}

lda = LatentDirichletAllocation()

#randomized grid search instead of a grid search? to save on time and computing power due to large dataset
model = RandomizedSearchCV(lda, search_params, n_iter=15, n_jobs=-1)

model.fit(word_vec)

joblib.dump(model, '../results/lda_randomgrid_model_10.joblib')
joblib.dump(word_vec, '../results/word_matrix_10.joblib')
joblib.dump(vectorizer, '../results/vectorizer_10.joblib')