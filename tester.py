import sys
from scipy.optimize import fmin
from forager.foraging import forage
from forager.switch import switch_delta, switch_multimodal, switch_simdrop, switch_troyer
from forager.cues import create_history_variables
from forager.utils import prepare_web_data, evaluate_web_data
import pandas as pd
import numpy as np
from scipy.optimize import fmin
import os
from tqdm import tqdm

normspath = 'data/norms/animals_snafu_scheme.csv'
similaritypath = 'data/lexical_data/similaritymatrix.csv'
frequencypath = 'data/lexical_data/frequencies.csv'
phonpath = 'data/lexical_data/phonmatrix.csv'

# Global Variables
models = ['static', 'dynamic', 'pstatic', 'pdynamic', 'all']
switch_methods = ['simdrop', 'multimodal', 'troyer', 'delta', 'all']

def retrieve_data(file):
    """
    1. Verify that data path exists. Make all truncations and replacements. 
    """
    data = prepare_web_data(file, delimiter='\t')
    return data

import pandas as pd
ls = retrieve_data('test files/test-file.txt')

norms = pd.read_csv('data/norms/animals_snafu_scheme.csv')
switch_troyer(ls[0][1],norms)