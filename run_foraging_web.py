# forager-web
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

#import tensorflow as tf

#import tensorflow_hub as hub
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
        print("inside norms_associative if statement")
        switch_names.append(switch_methods[3])
        switch_vecs.append(switch_norms_categorical(fluency_list, norms))
        print("back from switch_norms_associative")

    if switch == switch_methods[4] or switch == switch_methods[5]:
        for i, r in enumerate(rise):
            for j, f in enumerate(fall):
                switch_names.append(
                    "delta_rise={rise}_fall={fall}".format(rise=r, fall=f))
                switch_vecs.append(switch_delta(
                    fluency_list, semantic_similarity, r, f))
    print("back from all")

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
        print("back from calculate_switch")
    
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

def indiv_desc_stats(lexical_results, switch_results = None):
    metrics = lexical_results[['Subject', 'Semantic_Similarity', 'Frequency_Value', 'Phonological_Similarity']]
    metrics.replace(.0001, np.nan, inplace=True)
    grouped = metrics.groupby('Subject').agg(['mean', 'std'])
    grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
    grouped.reset_index(inplace=True)
    num_items = lexical_results.groupby('Subject')['Fluency_Item'].size()
    grouped['#_of_Items'] = num_items[grouped['Subject']].values
    # create column for each switch method per subject and get number of switches, mean cluster size, and sd of cluster size for each switch method
    if switch_results is not None:
        # count the number of unique values in the Switch_Method column of the switch_results DataFrame
        n_rows = len(switch_results['Switch_Method'].unique())
        new_df = pd.DataFrame(np.nan, index=np.arange(len(grouped) * (n_rows)), columns=grouped.columns)

        # Insert the original DataFrame into the new DataFrame but repeat the value in 'Subject' column n_rows-1 times

        new_df.iloc[(slice(None, None, n_rows)), :] = grouped
        new_df['Subject'] = new_df['Subject'].ffill()

        switch_methods = []
        num_switches_arr = []
        cluster_size_mean = []
        cluster_size_sd = []
        for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
            switch_method = sub[1]
            cluster_lengths = []
            num_switches = 0
            ct = 0
            for x in fl_list['Switch_Value'].values:
                ct += 1
                if x == 1:
                    num_switches += 1
                    cluster_lengths.append(ct)
                    ct = 0
            if ct != 0:
                cluster_lengths.append(ct)
            avg = sum(cluster_lengths) / len(cluster_lengths)
            sd = np.std(cluster_lengths)
            switch_methods.append(switch_method)
            num_switches_arr.append(num_switches)
            cluster_size_mean.append(avg)
            cluster_size_sd.append(sd)

        new_df['Switch_Method'] = switch_methods
        new_df['Number_of_Switches'] = num_switches_arr
        new_df['Cluster_Size_mean'] = cluster_size_mean
        new_df['Cluster_Size_std'] = cluster_size_sd
        grouped = new_df
        
    return grouped

def agg_desc_stats(switch_results, model_results=None):
    agg_df = pd.DataFrame()
    # get number of switches per subject for each switch method
    switches_per_method = {}
    for sub, fl_list in switch_results.groupby(["Subject", "Switch_Method"]):
        method = sub[1]
        if method not in switches_per_method:
            switches_per_method[method] = []
        if 1 in fl_list['Switch_Value'].values:
            switches_per_method[method].append(fl_list['Switch_Value'].value_counts()[1])
        else: 
            switches_per_method[method].append(0)
    agg_df['Switch_Method'] = switches_per_method.keys()
    agg_df['Switches_per_Subj_mean'] = [np.average(switches_per_method[k]) for k in switches_per_method.keys()]
    agg_df['Switches_per_Subj_SD'] = [np.std(switches_per_method[k]) for k in switches_per_method.keys()]
    
    if model_results is not None:
        betas = model_results.drop(columns=['Subject', 'Negative_Log_Likelihood_Optimized'])
        betas.drop(betas[betas['Model'] == 'forage_random_baseline'].index, inplace=True)
        grouped = betas.groupby('Model').agg(['mean', 'std'])
        grouped.columns = ['{}_{}'.format(col[0], col[1]) for col in grouped.columns]
        grouped.reset_index(inplace=True)

        # add a column to the grouped dataframe that contains the switch method used for each model
        grouped.loc[grouped['Model'].str.contains('static'), 'Model'] += ' none'
        # if the model name starts with 'forage_dynamic_', ''forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', or 'forage_phonologicaldynamicswitch_', replace the second underscore with a space
        switch_models = ['forage_dynamic_', 'forage_phonologicaldynamicglobal_', 'forage_phonologicaldynamiclocal_', 'forage_phonologicaldynamicswitch_']
        for model in switch_models:
            # replace only the second underscore with a space
            grouped.loc[grouped['Model'].str.contains(model), 'Model'] = grouped.loc[grouped['Model'].str.contains(model), 'Model'].str.replace('_', ' ', 2)
            grouped.loc[grouped['Model'].str.contains("forage "), 'Model'] = grouped.loc[grouped['Model'].str.contains("forage "), 'Model'].str.replace(' ', '_', 1)
        
        # split the Model column on the space
        grouped[['Model', 'Switch_Method']] = grouped['Model'].str.rsplit(' ', n=1, expand=True)

        # merge the two dataframes on the Switch_Method column 
        agg_df = pd.merge(agg_df, grouped, how='outer', on='Switch_Method')


    return agg_df
 