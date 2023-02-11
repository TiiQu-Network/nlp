from tiiqu import load_tiiqu_data, split
from preprocess import extract_word_list, \
  drop_stopwords, \
  extract_bigrams, \
  lemmatize, \
  get_dictionary ,\
  get_bow
from model_nmf import NMF

data = load_tiiqu_data( './data/nlpdata.csv' )
train_data, test_data = split( data )

wordlist = extract_word_list( train_data )
wordlist = drop_stopwords( wordlist )
bigrams = extract_bigrams( wordlist )
lemlists = lemmatize( bigrams )

dictionary = get_dictionary( bigrams )
corpus = get_bow( dictionary, bigrams )
model = NMF( corpus, num_topics=5, dic=dictionary )