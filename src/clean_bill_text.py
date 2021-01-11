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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
stopwords_ = set(stopwords.words('english'))
additional_stopwords = ('congress', 'act', 'states', 'united', 
                        'house', '116th', 'html', 'pre', 'body', 'doc',
                        'session', 'bill', 'introduced', 
                        'title','gt', 'subsection', 'paragraph', 'subparagraph',
                        'insert', 'section', 'mr', 'ms', 'mrs', 'shall', 'sec',
                        'lt', 'th', 'st', 'a', 'b', 'c', 'd' , 'e', 'f', 'g' , 'h' ,'res' ,'j')
roman_numerals = ('i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv')
stopwords_ = stopwords_.union(additional_stopwords)
stopwords_ = stopwords_.union(roman_numerals)

def stringy_soup(soup):
    '''
    takes in soup
    returns string
    lowercase
    remove numbers
    remove punctuation
    '''
    
    text = str(soup)
    #text = re.sub(r'(?<=DELETED)(.*)(?=\n&lt;DELETED)', '', text)
    text = re.sub(r'&lt;DELETED&gt[^>]+lt;/DELETED&gt', '', text)
    text = text.lower()
    text = re.sub('[0-9]+', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = word_tokenize(text)
    text = [w for w in text if w not in stopwords_]
    text = ' '.join(text)
    
    return text

nlp = spacy.load('en_core_web_md')
nlp.max_length = 20000000

def lemmatize(text):     
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)

version_def = {
 'ih': 'Introduced in House',
 'is': 'Introduced in Senate',
 'rh': 'Reported in (House) - version reported by the committee(s) including changes if any. It is then available for floor consideration',
 'rs': 'Reported in (Senate) - version reported by the committee(s) including changes if any. It is then available for floor consideration',
 'pch': 'Version placed on House calendar for consideration',
 'pcs': 'Version placed on the Senate calendar for consideration',
 'cph': 'Considered and passed by house',
 'cps': 'Considered and passed by Senate',
 'eh': 'Engrossed (House) - version that is the official copy as passed before it is sent to the Senate',
 'es': 'Engrossed (Senate) - version that is the official copy as passed before sent to House',
 'rdh': 'recieved in House from Senate',
 'rds': 'received in Senate from House',
 'rfh': 'Referred to House committee after being recieved from Senate',
 'rfs': 'Referred to Senate commidttee after being recieved from House',
 'rch': 'Reference change (House) - version re-referred to different or additional committee',
 'rcs': 'Reference change (Senate) - version re-referred to different or additional committee',
 'eah': 'Engrossed Amendment (House) - same as EH but often is the engrossment of an amendment which replaces the entire text of a measure',
 'eas': 'Engrossed Amendment (Senate) - same as ES but often is the engrossment of an amendment which replaces the entire text of a measure',
 'iph': 'Indefinitely postponed',
 'ips': 'Indefinitely postponed',
 'enr': 'Initial copy of a bill or joint resolution which has passed both houses in identical form'
}
ordered_versions = list(version_def.keys())
ordered_versions.reverse()
#not sure how to deal with amendments right now
ordered_versions.remove('eah')
ordered_versions.remove('eas')

bill_type = ['s', 'sjres', 'hr', 'hjres']
congress = [116, 115]

for c in congress:
    all_folders = []
    for folder in bill_type:
        rootdir = '../data/' + str(c) + '/bills_text/' + folder
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                all_folders.append(os.path.join(subdir, file))

    #the bill text are saved in the zipped folders
    zipped = [x for x in all_folders if '.zip' in x]
    # I only want to use the most recent version of the bill
    bill_num_versions = dict()
    for i in zipped:
        elem = i.split('/')
        bill_num = elem[5]
        vers = elem[7]
        if bill_num in bill_num_versions:
            bill_num_versions[bill_num].append(vers)
        else:
            bill_num_versions[bill_num] = []
            bill_num_versions[bill_num].append(vers)


    latest_version = dict()
    for bill, ver in bill_num_versions.items():
        for j in ordered_versions:
            if j in ver:
                latest_version[bill] = j
                break

    latest_version_zipped = []
    for k, v in latest_version.items():
        remove_digits = str.maketrans('','',string.digits)
        bill_type = k.translate(remove_digits)
        path = ('../data/' + str(c) + '/bills_text/' +
                bill_type + '/' + 
                k + '/text-versions/' +
                v + '/package.zip')
        latest_version_zipped.append(path)


    all_bills = {'bill_num':[],
                'type':[],
                'text':[]}

    for i in range(len(latest_version_zipped)):
        try:
            path_elems = latest_version_zipped[i].split('/')
            unzipped_folder = "BILLS-" + path_elems[2] + path_elems[5] + path_elems[7]
            html_file_path = unzipped_folder + "/html/" + unzipped_folder + '.htm'

            zf = zipfile.ZipFile(latest_version_zipped[i])
            file = zf.open(html_file_path)
            soup = BeautifulSoup(file, 'html.parser')
            
            all_bills['text'].append(lemmatize(stringy_soup(soup)))
            all_bills['bill_num'].append(path_elems[5])
            all_bills['type'].append(path_elems[7])
            
            print(f'saved {c} {path_elems[5]}')
            
        except Exception as e:
            print(f"{latest_version_zipped[i]} failed to process because {repr(e)}")

    bill_text_df = pd.DataFrame(all_bills)
    bill_text_df['congress'] = c

    csv_name = str(c) + 'bill_text.csv'
    bill_text_df.to_csv('../results/'+csv_name, index=False)