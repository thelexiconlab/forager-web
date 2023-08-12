#@title Load the Universal Sentence Encoder's TF Hub module

import tensorflow as tf

import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import re
from alive_progress import alive_bar 


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)


frequencypath = '../data/lexical_data/frequencies.csv'

labels = pd.read_csv(frequencypath, header=None)[0].values.tolist()

# loop through labels and get embeddings
embeddings = []
with alive_bar(len(labels)) as bar:
    for label in labels:
        embeddings.append(embed([label]).numpy()[0])
        bar()

# store embeddings in dataframe
df = pd.DataFrame(embeddings)
df.to_csv('../data/lexical_data/USE_semantic_embeddings.csv', index=False, header=False)