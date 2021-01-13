import numpy as np
import pandas as pd
import json
import zipfile
from bs4 import BeautifulSoup
import os
from collections import Counter
import re
import string
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Additional information about the bill
bill_type = ['s', 'sjres', 'hr', 'hjres']

bill_info_folders = []
for folder in bill_type:
    rootdir = 'data/116/bills/'+folder
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            bill_info_folders.append(os.path.join(subdir, file))

json_files = [x for x in bill_info_folders if 'data.json' in x]

bill_info = {}
for file in json_files:
    open_file = open(file)
    bill = json.load(open_file)
    bill_info[bill['bill_id']] = bill

bill_info_df = pd.DataFrame(bill_info)
bill_info_df = bill_info_df.T

bill_info_df = bill_info_df[['bill_id',
                             'cosponsors',
                             'sponsor',
                             'official_title',
                             'subjects',
                             'subjects_top_term']]

bill_num = []
for i in bill_info_df['bill_id']:
    num = i.split('-')
    bill_num.append(num[0])

bill_info_df['bill_num'] = bill_num
bill_info_df.to_csv('116bill_info.csv', index=False)

bill_info_df = pd.read_csv('116bill_info.csv')
bills_df = bill_info_df.merge(bill_text_df, how='outer', left_on='bill_num', right_on='bill_num')

bills_df.to_csv('116bills.csv')