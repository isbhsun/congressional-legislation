import re

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from nltk.corpus import stopwords
from scipy import stats
import joblib

bill_text_df = pd.read_csv('../results/116bill_text.csv')

vectorizer = CountVectorizer(max_df=0.85,
                             min_df=5, 
                             max_features=5000)

word_vec = vectorizer.fit_transform(bill_text_df.text)

# search_params = {'n_components': [10, 15, 20, 25, 30, 35, 40],
#                  'learning_decay': [.5, .6, .7, .8, .9],
#                  'doc_topic_prior': [0.01, 0.025, 0.05, 0.075, 0.1, None],
#                  'topic_word_prior': [0.01, 0.025, 0.05, 0.075, 0.1, None],
#                  'learning_method': ['batch', 'online'],
#                  'n_jobs': [-1],
#                  'random_state': [0, 1, 2]}
search_params = {'n_components': [10, 15],
                 'learning_decay': [.5, .6],
                 'learning_method': ['online'],
                 'n_jobs': [-1],
                 'random_state': [0]}

lda = LatentDirichletAllocation()

#randomized grid search instead of a grid search? to save on time and computing power due to large dataset
model = GridSearchCV(lda, param_grid=search_params)

model.fit(word_vec)

joblib.dump(lda, '../results/lda_grid_model.joblib')
joblib.dump(vectorizer, '../results/vec.joblib')