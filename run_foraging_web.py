# forager-web
# Contains functions from run_foraging rewritten to support API requests in the api_router module.
# Changes:
# - retrieve_data() accepts a file object, not path.
# - All run_...() functions return dataframe(s) instead of saving results to file path.
# - run_model() is replaced by run_all()
# - Added run_switch() and run_model_nll() containing partitioned functionality of run_model()
# - Partitioned synthesize_results() into synthesize_all() with individual synthesize methods for switch, model, and nll

from forager.switch import switch_delta, switch_multimodal, switch_simdrop, switch_norms_associative, switch_norms_categorical, switch_slope_difference, switch_pei, cluster_lengths
from forager.cues import create_history_variables, get_oov_sims
from forager.utils import evaluate_web_data
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

#import tensorflow as tf
#import tensorflow_hub as hub

"""
"""

# Global Path Variabiles

normspath = 'data/norms/animals_snafu_scheme_vocab.csv'
similaritypath = 'data/lexical_data/USE_semantic_matrix.csv'
frequencypath = 'data/lexical_data/USE_frequencies.csv'
phonpath = 'data/lexical_data/USE_phonological_matrix.csv'

# Global Variables
switch_methods = ['simdrop','multimodal','norms_associative', 'norms_categorical', 'delta', 'slope_difference', 'pei', 'all']

# Parameters required by each parametric switch method when param_mode == 'specific'
specific_params = {'multimodal': ['alpha'], 'delta': ['rise', 'fall'], 'pei': ['alpha', 'beta', 'prior']}

# Methods


def get_evaluation_message(file, oov_choice='exclude', time_type='cumulative', time_units='s'):
    """
    Returns text feedback on replacements and truncations that would be made within data.
    Returns error message if evaluation was not successful.
    """
    message = ""

    try:

        data_df, replacement_df, data_lists = evaluate_web_data(file, oov_choice=oov_choice, time_type=time_type, time_units=time_units)
        
    
        exclude_count = (replacement_df["evaluation"] == "EXCLUDE").sum()    
        unk_count = (replacement_df["evaluation"] == "UNK").sum()        
        trunc_count = (replacement_df["evaluation"] == "TRUNCATE").sum()        
        replacement_count = (replacement_df["evaluation"] == "REPLACE").sum()
        

        # inform user about detected columns
        detected_cols = list(data_df.columns)
        if 'timepoint' in detected_cols and 'time' in detected_cols:
            message += "We detected columns: SID, entry, timepoint, and time (timing data). \n "
        elif 'time' in detected_cols:
            message += "We detected columns: SID, entry, and time (timing data). \n "
        elif 'timepoint' in detected_cols:
            message += "We detected columns: SID, entry, and timepoint. \n "
        
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

def calculate_switch(switch, fluency_list, semantic_similarity, phon_similarity, norms, times=None, alpha=np.arange(0, 1.1, 0.1), rise=np.arange(0, 1.25, 0.25), fall=np.arange(0, 1.25, 0.25), pei_beta=np.arange(0, 1.1, 0.1), pei_prior=np.arange(0.1, 1.0, 0.1), return_probabilities=False):
    '''
    1. Check if specified switch model is valid
    2. Return set of switches, including parameter value, if required
    switch_methods = ['simdrop','multimodal','norms_associative', 'norms_categorical', 'delta', 'slope_difference', 'pei', 'all']

    return_probabilities: if True, also return a list aligned with switch_vecs holding the PEI
        posterior P(cluster) array for PEI vectors and None for other methods.
    '''
    switch_names = []
    switch_vecs = []
    switch_probs = []

    if switch not in switch_methods:
        ex_str = "Specified switch method is invalid. Switch method must be one of the following: {switch}".format(
            switch=switch_methods)
        raise Exception(ex_str)

    run_all = switch == 'all'

    if switch == 'simdrop' or run_all:
        switch_names.append('simdrop')
        switch_vecs.append(switch_simdrop(fluency_list, semantic_similarity))

    if switch == 'multimodal' or run_all:
        for i, a in enumerate(alpha):
            switch_names.append('multimodal_alpha={alpha}'.format(alpha=a))
            switch_vecs.append(switch_multimodal(
                fluency_list, semantic_similarity, phon_similarity, a))

    if switch == 'norms_associative' or run_all:
        switch_names.append('norms_associative')
        switch_vecs.append(switch_norms_associative(fluency_list, norms))

    if switch == 'norms_categorical' or run_all:
        switch_names.append('norms_categorical')
        switch_vecs.append(switch_norms_categorical(fluency_list, norms))

    if switch == 'delta' or run_all:
        for i, r in enumerate(rise):
            for j, f in enumerate(fall):
                switch_names.append(
                    "delta_rise={rise}_fall={fall}".format(rise=r, fall=f))
                switch_vecs.append(switch_delta(
                    fluency_list, semantic_similarity, r, f))

    # Precompute slope differences once (used by slope_difference, PEI, and lexical output)
    precomputed_slope_diffs = None
    if times is not None:
        _, precomputed_slope_diffs = switch_slope_difference(fluency_list, times)

    if switch == 'slope_difference' or run_all:
        if times is not None:
            switch_names.append("slope_difference")
            decisions = [2]
            for i in range(len(precomputed_slope_diffs)):
                decisions.append(1 if precomputed_slope_diffs[i] < 0 else 0)
            switch_vecs.append(decisions)

    if switch == 'pei' or run_all:
        if times is not None:
            for i, a in enumerate(alpha):
                for j, b in enumerate(pei_beta):
                    for k, p in enumerate(pei_prior):
                        switch_names.append("pei_alpha={alpha}_beta={beta}_prior={prior}".format(alpha=a, beta=b, prior=p))
                        decisions, probs = switch_pei(fluency_list, times, semantic_similarity, phon_similarity, slope_diffs=precomputed_slope_diffs, alpha=a, beta=b, prior_probability=p, return_probabilities=True)
                        switch_vecs.append(decisions)
                        switch_probs.append(probs)

    if return_probabilities:
        # PEI vectors are appended last; all other methods have no probabilities
        switch_probs = [None] * (len(switch_vecs) - len(switch_probs)) + switch_probs
        return switch_names, switch_vecs, precomputed_slope_diffs, switch_probs
    return switch_names, switch_vecs, precomputed_slope_diffs


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
    #print("Running oov sims")
    print(data)
    
    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
    outputs = []

    # Run through each fluency list in dataset
    for entry in tqdm(data):
        subj, fl_list = entry[0], entry[1]
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
    for entry in tqdm(data):
        subj, fl_list = entry[0], entry[1]
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

def check_param_values(switch, param_mode, param_values):
    """
    Returns an error message if the parameter values needed by the selected switch method in
    'specific' mode are missing or out of range, or None if they are valid.
    alpha, beta, rise and fall must be in [0, 1]; prior must be in [0.1, 0.9] (the range of the grid).
    Grid mode and methods without parameters (including 'all', which always uses the grids) need no values.
    """
    if param_mode not in ('grid', 'specific'):
        return "Invalid parameter mode '{}': use 'grid' or 'specific'.".format(param_mode)
    if param_mode == 'grid' or switch not in specific_params:
        return None
    param_values = param_values or {}
    for key in specific_params[switch]:
        value = param_values.get(key)
        if value is None:
            return "Missing value for parameter '{}', which the {} method needs when specific values are selected.".format(key, switch)
        if key == 'prior':
            if not 0.1 <= value <= 0.9:
                return "Parameter 'prior' must be between 0.1 and 0.9 (got {}).".format(value)
        elif not 0 <= value <= 1:
            return "Parameter '{}' must be between 0 and 1 (got {}).".format(key, value)
    return None

def run_switch(data, switch, param_mode='grid', param_values=None):
    """
    Perform only switch computations.
    Outputs a dataframe for switch results.
    In 'specific' mode, the values needed by the selected method must be present in param_values
    (see check_param_values); otherwise a ValueError is raised.
    """

    error = check_param_values(switch, param_mode, param_values)
    if error is not None:
        raise ValueError(error)

    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()

    # Default grids (rounded to remove floating point noise, e.g. 0.30000000000000004)
    alpha = np.round(np.arange(0, 1.1, 0.1), 10)
    rise = np.round(np.arange(0, 1.25, 0.25), 10)
    fall = np.round(np.arange(0, 1.25, 0.25), 10)
    pei_beta = np.round(np.arange(0, 1.1, 0.1), 10)
    pei_prior = np.round(np.arange(0.1, 1.0, 0.1), 10)

    # Specific values replace the grids of the parameters used by the selected method
    if param_mode == 'specific' and switch in specific_params:
        if 'alpha' in specific_params[switch]:
            alpha = np.array([param_values['alpha']])
        if switch == 'delta':
            rise = np.array([param_values['rise']])
            fall = np.array([param_values['fall']])
        if switch == 'pei':
            pei_beta = np.array([param_values['beta']])
            pei_prior = np.array([param_values['prior']])

    switch_results = pd.DataFrame()
    lexical_results = pd.DataFrame()
    for entry in tqdm(data):
        # Unpack with optional timing data
        if len(entry) == 3:
            subj, fl_list, times = entry
        else:
            subj, fl_list = entry
            times = None

        history_vars = create_history_variables(fl_list, labels, similarity_matrix, frequency_list, phon_matrix)

        ## create lexical results dataframe

        lexical_df = pd.DataFrame()
        lexical_df['Subject'] = len(fl_list) * [subj]
        lexical_df['Fluency_Item'] = fl_list
        lexical_df['Semantic_Similarity'] = history_vars[0]
        lexical_df['Frequency_Value'] = history_vars[2]
        lexical_df['Phonological_Similarity'] = history_vars[4]

        # Add timing columns if timing data is available
        if times is not None:
            cumulative = np.array(times)
            lexical_df['Cumulative_IRT'] = cumulative
            irt = np.empty(len(cumulative))
            irt[0] = cumulative[0]
            irt[1:] = np.diff(cumulative)
            lexical_df['IRT'] = irt

        # history_vars contains sim_list, sim_history, freq_list, freq_history,phon_list, phon_history
        switch_names, switch_vecs, slope_diffs, switch_probs = calculate_switch(switch, fl_list, history_vars[0], history_vars[4], norms, times=times,
                                                     alpha=alpha, rise=rise, fall=fall, pei_beta=pei_beta, pei_prior=pei_prior,
                                                     return_probabilities=True)

        # Add slope differences to lexical results (from precomputed values)
        if times is not None:
            sd_col = np.empty(len(fl_list))
            sd_col[0] = np.nan
            if slope_diffs is not None and len(slope_diffs) > 0:
                sd_col[1:] = slope_diffs
            else:
                sd_col[1:] = np.nan
            lexical_df['Slope_Difference'] = sd_col

        lexical_results = pd.concat([lexical_results, lexical_df], ignore_index=True)

        ## create switch results dataframe

        switch_df = pd.DataFrame()
        for j, switch_val in enumerate(switch_vecs):
            df = pd.DataFrame()
            df['Subject'] = len(switch_val) * [subj]
            df['Fluency_Item'] = fl_list
            df['Switch_Value'] = switch_val
            df['Switch_Method'] = switch_names[j]
            # posterior P(cluster) for PEI rows, NaN for other methods
            df['PEI_Cluster_Probability'] = switch_probs[j] if switch_probs[j] is not None else np.nan
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
    metrics = lexical_results[['Subject', 'Semantic_Similarity', 'Frequency_Value', 'Phonological_Similarity']].copy()
    # the first item of each list has placeholder similarities (.0001); other .0001 values are real (e.g. no phonological overlap)
    metrics.loc[metrics.groupby('Subject').head(1).index, ['Semantic_Similarity', 'Phonological_Similarity']] = np.nan
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
            switch_values = fl_list['Switch_Value'].values
            lengths = cluster_lengths(switch_values)
            num_switches = int((switch_values == 1).sum())
            avg = sum(lengths) / len(lengths)
            sd = np.std(lengths)
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
 