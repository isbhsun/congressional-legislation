import zipfile
from bs4 import BeautifulSoup
import re
import string
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import gensim
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
stopwords_ = set(stopwords.words('english'))
additional_stopwords = ('congress', 'act', 'states', 'united', 
                        'house', '116th', 'hundred', 'sixteenth', 'html', 'pre', 'body', 'doc',
                        'session', 'bill', 'introduced', 'two', 'thousand', 'twenty', 
                        'title','gt', 'subsection', 'paragraph', 'subparagraph',
                        'insert', 'section', 'mr', 'ms', 'mrs', 'shall', 'sec', 'law', 'year', 'secretary',
                        'semicolon', 'comma', 'include', 'dz', 'af', 'etc', 'llc',
                        'lt', 'th', 'st', 'a', 'b', 'c', 'd' , 'e', 'f', 'g' , 'h' ,'res' ,'j', 'r',
                        'aa', 'bb', 'cc', 'de', 'dt', 'dz', 'ee', 'ff','gg','hh','kk','mm','nn','oo','ss','ww')
roman_numerals = ('i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi',
                  'xii', 'xiii', 'xiv', 'xix', 'xv', 'xvi', 'xvii', 'xx', 'xxi', 'xxii', 'xviii','xxiii','xxvii')
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

#Bigram function 
def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
