# Contains functions from run_foraging rewritten to support API requests in the api_router module.
# Changes:
# - retrieve_data() accepts a file object, not path.
# - All run_...() functions return dataframe(s) instead of saving results to file path.
# - run_model() is replaced by run_all()
# - Added run_switch() and run_model_nll() containing partitioned functionality of run_model()
# - Partitioned synthesize_results() into synthesize_all() with individual synthesize methods for switch, model, and nll

import sys
from scipy.optimize import fmin
from forager.foraging import forage
from forager.switch import switch_delta, switch_multimodal, switch_simdrop, switch_norms_associative, switch_norms_categorical
from forager.cues import create_history_variables, get_oov_sims
from forager.utils import evaluate_web_data
import pandas as pd
import numpy as np
from scipy.optimize import fmin
import os
from tqdm import tqdm

import tensorflow as tf

import tensorflow_hub as hub
import re
from alive_progress import alive_bar 

"""
"""

# Global Path Variabiles

normspath = 'data/norms/animals_snafu_scheme_vocab.csv'
similaritypath = 'data/lexical_data/USE_semantic_matrix.csv'
frequencypath = 'data/lexical_data/USE_frequencies.csv'
phonpath = 'data/lexical_data/USE_phonological_matrix.csv'

# Global Variables
switch_methods = ['simdrop','multimodal','norms_associative', 'norms_categorical', 'delta','all']

# Methods


def get_evaluation_message(file, oov_choice='exclude'):
    """
    Returns text feedback on replacements and truncations that would be made within data. 
    Returns error message if evaluation was not successful. 
    """
    message = ""

    try:

        data_df, replacement_df, data_lists = evaluate_web_data(file, oov_choice = oov_choice)
        
    
        exclude_count = (replacement_df["evaluation"] == "EXCLUDE").sum()    
        unk_count = (replacement_df["evaluation"] == "UNK").sum()        
        trunc_count = (replacement_df["evaluation"] == "TRUNCATE").sum()        
        replacement_count = (replacement_df["evaluation"] == "REPLACE").sum()
        

        # if data_df has 3 columns, then it has timepoint, tell them that
        if (len(data_df.columns) == 3):
            message += "Your data has been evaluated. We found 3 columns in your data, so we will treat the first column as the subject ID, the second column as the fluency list, and the third column as the timepoint. \n "
        
        if (replacement_count + trunc_count + exclude_count + unk_count == 0):
            message += "Congrats! We have found all items from your data in our vocabulary."
        else:
        
            message += "We found reasonable replacements for " + str(replacement_count)+ " item(s) in your data. \n\nAdditionally, "

            if(oov_choice == 'exclude'):
                message += "you chose to " + oov_choice + " OOV items."
                message += " We found " + str(exclude_count) + " such instance(s) across all lists."
            elif(oov_choice == 'truncate'):
                message += "you chose to " + oov_choice + " the list after any OOV items. "
                message += " We found " + str(trunc_count) + " such instance(s) across all lists."
            else:
                # oov_choice == 'random'
                message += "you chose to assign a random vector to any OOV items."            
                message += " We found " + str(unk_count) + " such item(s) and assigned them a random vector, across all lists."

        message += "\n\nThe data set after evaluation AND the dataset that will be used for analysis are available for download by clicking the button below."

    except Exception as e:
        message = "Error while evaluating data. Please check that your file is properly formatted."
    
    return message, replacement_df, data_df, data_lists

def get_lexical_data():
    norms = pd.read_csv(normspath, encoding="unicode-escape")
    similarity_matrix = np.loadtxt(similaritypath, delimiter=',')
    frequency_list = np.array(pd.read_csv(frequencypath, header=None, encoding="unicode-escape")[1])
    phon_matrix = np.loadtxt(phonpath, delimiter=',')
    labels = pd.read_csv(frequencypath, header=None)[0].values.tolist()
    return norms, similarity_matrix, phon_matrix, frequency_list, labels

def calculate_switch(switch, fluency_list, semantic_similarity, phon_similarity, norms, alpha=np.arange(0, 1.1, 0.1), rise=np.arange(0, 1.25, 0.25), fall=np.arange(0, 1.25, 0.25)):
    '''
    1. Check if specified switch model is valid
    2. Return set of switches, including parameter value, if required
    switch_methods = ['simdrop','multimodal','norms_associative', 'norms_categorical', 'delta','all']
    '''
    switch_names = []
    switch_vecs = []

    if switch not in switch_methods:
        ex_str = "Specified switch method is invalid. Switch method must be one of the following: {switch}".format(
            switch=switch_methods)
        raise Exception(ex_str)

    if switch == switch_methods[0] or switch == switch_methods[5]:
        switch_names.append(switch_methods[0])
        switch_vecs.append(switch_simdrop(fluency_list, semantic_similarity))

    if switch == switch_methods[1] or switch == switch_methods[5]:
        for i, a in enumerate(alpha):
            switch_names.append('multimodal_alpha={alpha}'.format(alpha=a))
            switch_vecs.append(switch_multimodal(
                fluency_list, semantic_similarity, phon_similarity, a))

    if switch == switch_methods[2] or switch == switch_methods[5]:
        print("inside norms_associative if statement")
        switch_names.append(switch_methods[2])
        switch_vecs.append(switch_norms_associative(fluency_list, norms))
        print("back from switch_norms_associative")
    
    if switch == switch_methods[3] or switch == switch_methods[5]:
        switch_names.append(switch_methods[3])
        switch_vecs.append(switch_norms_categorical(fluency_list, norms))

    if switch == switch_methods[4] or switch == switch_methods[5]:
        for i, r in enumerate(rise):
            for j, f in enumerate(fall):
                switch_names.append(
                    "delta_rise={rise}_fall={fall}".format(rise=r, fall=f))
                switch_vecs.append(switch_delta(
                    fluency_list, semantic_similarity, r, f))

    return switch_names, switch_vecs


def output_results(results, dname, dpath='output', sep=','):
    if os.path.exists(dpath) == False:
        os.mkdir(dpath)
    results[0].to_csv(os.path.join(
        dpath, dname + '_modelresults.csv'), index=False, sep=sep)
    results[1].to_csv(os.path.join(
        dpath, dname + '_switchresults.csv'), index=False, sep=sep)
    results[2].to_csv(os.path.join(
        dpath, dname + '_individualitemfits.csv'), index=False, sep=sep)

def run_sims_oov(data):

    """
    Perform only switch computations.   
    Outputs a dataframe for switch results. 
    """
    print("Running oov sims")
    print(data)
    
    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
    outputs = []

    # Run through each fluency list in dataset
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        print(fl_list)
        # Get History Variables
        history_vars = get_oov_sims(
            fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        
        outputs.append([subj, fl_list, history_vars])

    return synthesize_sim_results(outputs)


def run_sims(data):

    """
    Perform only switch computations.   
    Outputs a dataframe for switch results. 
    """
    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
    lexical_results = []
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        # history_vars contains sim_list, sim_history, freq_list, freq_history,phon_list, phon_history
        lexical_df = pd.DataFrame()
        lexical_df['Subject'] = len(fl_list) * [subj]
        lexical_df['Fluency_Item'] = fl_list
        lexical_df['Semantic_Similarity'] = history_vars[0]
        lexical_df['Frequency_Value'] = history_vars[2]
        lexical_df['Phonological_Similarity'] = history_vars[4]
        lexical_results.append(lexical_df)
    lexical_results = pd.concat(lexical_results,ignore_index=True)
    return lexical_results

def run_switch(data, switch):
    """
    Perform only switch computations.   
    Outputs a dataframe for switch results. 
    """
    
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
    switch_results = pd.DataFrame()
    lexical_results = pd.DataFrame()
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        
        history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        
        ## create lexical results dataframe

        lexical_df = pd.DataFrame()
        lexical_df['Subject'] = len(fl_list) * [subj]
        lexical_df['Fluency_Item'] = fl_list
        lexical_df['Semantic_Similarity'] = history_vars[0]
        lexical_df['Frequency_Value'] = history_vars[2]
        lexical_df['Phonological_Similarity'] = history_vars[4]

        lexical_results = pd.concat([lexical_results, lexical_df], ignore_index=True)
        # history_vars contains sim_list, sim_history, freq_list, freq_history,phon_list, phon_history
        switch_names, switch_vecs = calculate_switch(switch, fl_list, history_vars[0], history_vars[4], norms)
    
        ## create switch results dataframe
    
        switch_df = pd.DataFrame()
        for j, switch_val in enumerate(switch_vecs):
            df = pd.DataFrame()
            df['Subject'] = len(switch_val) * [subj]
            df['Fluency_Item'] = fl_list
            df['Switch_Value'] = switch_val
            df['Switch_Method'] = switch_names[j]
            switch_df = pd.concat([switch_df, df], ignore_index=True)
    
        switch_results = pd.concat([switch_results, switch_df], ignore_index=True)
    
    return switch_results, lexical_results

def synthesize_sim_results(outputs):

    """
    Returns dataframe containing results of switch computations. 
    Params:
    - Array of [subject, fluency list, switch names, and switch vectors] 
    Output:
    - Switch Vector Result(s)
    """

    sim_results = []

    for output in outputs:
        subj = output[0]
        fl_list = output[1]    
        df = pd.DataFrame()
        df['Subject'] = len(fl_list) * [subj]
        df['Fluency_Item'] = fl_list
        # # add lexical metrics to switch results
        df['Semantic_Similarity'] = output[2][0]
        df['Frequency_Value'] = output[2][1]
        df['Phonological_Similarity'] = output[2][2]

        sim_results.append(df)

    sim_results = pd.concat(sim_results, ignore_index=True)

    return sim_results


def synthesize_switch_results(outputs):
    """
    Returns dataframe containing results of switch computations. 
    Params:
    - Array of [subject, fluency list, switch names, and switch vectors] 
    Output:
    - Switch Vector Result(s)
    """

    switch_results = []

    for output in outputs:
        subj = output[0]
        fl_list = output[1]
        switch_methods = output[2]
        switch_vectors = output[3]

        # history vars returns sim_list, freq_list, phon_list,sim_history, freq_history, phon_history
        # it is contained in output[4]

        # Create  Switch Results DataFrame
        switch_df = []
        for j, switch in enumerate(switch_vectors):
            df = pd.DataFrame()
            df['Subject'] = len(switch) * [subj]
            df['Fluency_Item'] = fl_list
            df['Switch_Value'] = switch
            df['Switch_Method'] = switch_methods[j]
            # # add lexical metrics to switch results
            df['Semantic_Similarity'] = output[4][0]
            df['Frequency_Value'] = output[4][1]
            df['Phonological_Similarity'] = output[4][2]
            switch_df.append(df)
        
        switch_df = pd.concat(switch_df, ignore_index=True)
        switch_results.append(switch_df)

    switch_results = pd.concat(switch_results, ignore_index=True)

    return switch_results