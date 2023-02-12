import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
spC = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
postags = ['NOUN','VERB', 'ADJ', 'ADV'] # Keep nouns, adj, verbs
from gensim.parsing.porter import PorterStemmer
p = PorterStemmer()

#converting dataframe type to word list for further pre-processing stages
def extract_word_list(df):
    doc_list = df.text.values.tolist()
    word_list = [gensim.utils.simple_preprocess(txt, deacc=True, min_len=3) for txt in doc_list]
    return word_list

#dropping the stopwords could be helpful to keep more informative words
def drop_stopwords(word_list):
    nltk.download('stopwords', quiet=True)
    st_words = stopwords.words('english')
    st_words.extend(['from', 'subject', 're', 'edu', 'use'])
    word_list_nostops = [[word for word in txt if word not in st_words] for txt in word_list]
    return word_list_nostops


def extract_bigrams(word_list):
    bigram = Phrases(word_list, min_count=5, threshold=100)
    bigram_model = Phraser(bigram)
    word_bigrams = [bigram_model[w_vec] for w_vec in word_list]
    return word_bigrams



def lemmatize(word_bigrams, ptags = postags, spC=spC):
    '''Lemmatizes words based on allowed postags, input format is list of sublists 
       with strings'''
       
    lem_lists =[]
    for vec in word_bigrams:
        sentence = spC(" ".join(vec))
        lem_lists.append([token.lemma_ for token in sentence if token.pos_ in ptags])
    
    return lem_lists

def stemming(word_bigrams, porter_stemmer=p):
    stem_lists =[]
    for vec in word_bigrams:
        stem_lists.append([p.stem(token) for token in vec])
    
    return stem_lists

# In gensim a dictionary is a mapping between words and their integer id
def get_dictionary(word_bigrams):
    dictionary = Dictionary(word_bigrams)
    # Filter out extremes to limit the number of features
    dictionary.filter_extremes(no_below=3, no_above=0.85, keep_n=5000)
    return dictionary

# Create the bag-of-words format (list of (token_id, token_count))
def get_bow(dic, word_bigrams):
    corpus = [dic.doc2bow(text) for text in word_bigrams]
    return corpus
def tfidf_rep(word_list):
    vect = TfidfVectorizer(min_df=50, stop_words='english')
 
    # Fit and transform
    X = vect.fit_transform(word_list)
    return X
