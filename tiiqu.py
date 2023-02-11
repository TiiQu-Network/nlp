'''
This module aims to loading, running basic cleansing and splitting of tiiqu dataset to train and test subsets.
'''
#importing relavant libraries
import numpy as np
import pandas as pd
import re
import seaborn as sns
import sklearn as sk
import sklearn.model_selection

def load_tiiqu_data(filepath):
    df = pd.read_csv(filepath, encoding = 'utf8')
    #dropping special charachters
    for col in df.columns:
        df[col] = df[col].apply(lambda s: re.sub(r' *[^\x00-\x7F]+', '', s ) if pd.notnull(s) else s).astype('string')
        df[col] = df[col].apply(lambda s: re.sub(r'\S*@\S*\s?', '', s) if pd.notnull(s) else s).astype('string')
        df[col] = df[col].apply(lambda s: re.sub(r"\'", "", s) if pd.notnull(s) else s).astype('string')
        df[col] = df[col].apply(lambda s: re.sub(r'\s+', ' ', s) if pd.notnull(s) else s).astype('string')
    #filling missing question using previously asked questions
    df['Question'] = df.Question.fillna(method='ffill')
    #filling missing answers by null text
    df['Answer'] = df.Answer.fillna('')
    #making the Topic content lowercase
    #removing whitespaces from both sides of the content in "Topic" column of the dataframe
    df['Topic'] = df.Topic.apply(str.lower)
    df['Topic'] = df.Topic.apply(str.strip)
    #adding topic id to the columns of the dataset
    df['tpc_id'] = df.groupby('Topic', sort=False).ngroup()
    #shuffling the dataset rows in order to have more random arrangment of data throughout the records
    df = df.sample(frac=1).reset_index(drop=True)
    df['text'] = df['Page'].apply(str) + ' ' + df['Question'].apply(str) + df['Answer'].apply(str)
    df = df[['tpc_id', 'text']]
    return df

def split(df):
    kf = sklearn.model_selection.StratifiedKFold(n_splits = 5, shuffle = True)
    for i, (train_index, test_index) in enumerate(kf.split(df.text, df.tpc_id)):
        train = df.iloc[train_index]
        test = df.iloc[test_index]
        print('Fold %d : #Train items = %d, \t #Test items = %d' % (i+1, len(train), len(test)))
    return train, test