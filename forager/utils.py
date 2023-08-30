import numpy as np
from scipy import stats
import pandas as pd

import difflib
import nltk

def trunc(word, df):
    # function to truncate fluency list at word
    i = df[df['entry'] == word].index.values[0]
    sid = df.iloc[i]['SID']
    sid_rows = df[df['SID'] == sid].index.values
    j = sid_rows[-1] + 1
    df.drop(df.index[i:j], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return None

def exclude(word,df):
    # function to exclude all instances of word from df
    df.drop(df[df['entry'] == word].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return None

# Modified web version of prepareData which returns list of replacements and truncations that would be made using default policy.
def evaluate_web_data(file, delimiter = '\t', oov_choice = 'exclude'):
   
    ### LOAD BEHAVIORAL DATA ###
    df = pd.read_csv(file, header=None, names=['SID', 'entry'], delimiter=delimiter)
    replacement_df = df.copy()
    # load labels
    labels = pd.read_csv("data/lexical_data/frequencies.csv", names=['word', 'logct', 'ct']) 

    # set all replacements to actual word for all words in labels as the default
    replacements = {word: word for word in labels['word'].values}

    # get values from df 
    values = df['entry'].values
    
    # loop through values to find which ones are not in file
    oov = [w for w in values if w not in labels['word'].values]
    
    if len(oov) > 0:
        for word in set(oov):
                # get closest match in vocab and check edit distance
                closest_word = difflib.get_close_matches(word, labels['word'].values,1)

                if len(closest_word)>0 and nltk.edit_distance(word, closest_word[0]) <= 2:
                    replacements[word] = closest_word[0]
                elif oov_choice == "exclude":
                    # exclude this word from the list
                    exclude(word, df)
                    replacements[word] = "EXCLUDE"
                elif oov_choice == "random":
                    # change all occurrences of word to "UNK"
                    replacements[word] = "UNK"
                elif oov_choice == "process":
                    # if they want to create embeddings on the fly, do that here
                    replacements[word] = word
                else: 
                    # truncate fluency list before instance of OOV item
                    while word in df.values:
                        trunc(word, df)
                    replacements[word] = "TRUNCATE"
                df.replace(replacements, inplace=True)

    # add an extra column to orig_df with the replacement word

    replacement_df['evaluation'] = replacement_df['entry'].map(replacements)
    # create a new column 'replacement' that is a copy of 'evaluation'
    replacement_df['replacement'] = replacement_df['evaluation']
    # now replace all instances in evalution where the entry doesn't match the replacement AND isn't within
    # ['UNK', 'EXCLUDE', 'TRUNCATE'] with 'REPLACE'
    replacement_df.loc[(replacement_df['entry'] != replacement_df['evaluation']) & (~replacement_df['evaluation'].isin(['UNK', 'EXCLUDE', 'TRUNCATE'])), 'evaluation'] = 'REPLACE'
    # also for the column 'evaluation', if entry matches evaluation, replace with 'found'
    replacement_df.loc[(replacement_df['entry'] == replacement_df['evaluation']), 'evaluation'] = 'FOUND'

    # Stratify data into fluency lists
    data = []
    for subj in df['SID'].unique():
        subj_df = df[df['SID'] == subj]
        subj_data = (subj,subj_df['entry'].values.tolist())
        data.append(subj_data)
    
    return df, replacement_df, data
