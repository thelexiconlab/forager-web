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
from forager.switch import switch_delta, switch_multimodal, switch_simdrop, switch_troyer
from forager.cues import create_history_variables
from forager.utils import evaluate_web_data
import pandas as pd
import numpy as np
from scipy.optimize import fmin
import os
from tqdm import tqdm

"""
Workflow: 
1. Validate input(s)
    a. "Prepare Data" - does this also get required freq/sim data?
        - takes path of fluency list ; replace/truncated fluency list
2. Run model(s)
    a. Model Optimization: Currently, the code base doesn't do optimization implicity. We have to include that now.
        Question: Do we want to do the same and report optimized & unoptimized model results?; along with param values?
    b. Running through switch method(s)
3. Outputs:
    a. Results
    b. Optimized Parameters
    c. Runtime
    d. Best model(s)/switching?
4. Extras & Reporting/Comparison?:
    a. visualization(s)
    X b. statistical test(s) & reporting
"""

# Global Path Variabiles
#normspath = 'data/norms/troyernorms.csv'
normspath = 'data/norms/animals_snafu_scheme.csv'
similaritypath = 'data/lexical_data/similaritymatrix.csv'
frequencypath = 'data/lexical_data/frequencies.csv'
phonpath = 'data/lexical_data/phonmatrix.csv'

# Global Variables
models = ['static', 'dynamic', 'pstatic', 'pdynamic', 'all']
switch_methods = ['simdrop', 'multimodal', 'troyer', 'delta', 'all']

# Methods


def get_evaluation_message(file, oov_choice='exclude'):
    """
    Returns text feedback on replacements and truncations that would be made within data. 
    Returns error message if evaluation was not successful. 
    """
    message = ""

    try:

        data_df, replacement_df, data_lists = evaluate_web_data(file, delimiter='\t', oov_choice = oov_choice)
        
        exclude_count = (replacement_df["evaluation"] == "EXCLUDE").sum()
        unk_count = (replacement_df["evaluation"] == "UNK").sum()
        trunc_count = (replacement_df["evaluation"] == "TRUNCATE").sum()
        replacement_count = (replacement_df["evaluation"] == "REPLACE").sum()
        
        if (replacement_count + trunc_count + exclude_count + unk_count == 0):
            message = "Congrats! We have found all items from your data in our vocabulary. Please click the button below to get your results."
            return message, replacement_df
        
        message += "We have found reasonable replacements for " + str(replacement_count)+ " item(s) in your data. \n\nAdditionally, "

        if(oov_choice == 'exclude'):
            message += "you chose to " + oov_choice + " OOV items."
            message += " We found " + str(exclude_count) + " such instance(s) across all lists."
        elif(oov_choice == 'truncate'):
            message += "you chose to " + oov_choice + " the list after any OOV items. "
            message += " We found " + str(trunc_count) + " such instance(s) across all lists."
        else:
            # oov_choice == 'random'
            message += "you chose to assign a random vector to any OOV items."
            # count how many replacement's keys are UNK
            
            message += " We found " + str(unk_count) + " such item(s) and assigned them a random vector, across all lists."

        message += "\n\nThe data set after evaluation AND the dataset that will be used for analysis are available for download by clicking the button below."

    except Exception as e:
        message = "Error while evaluating data. Please check that your file is properly formatted."

    return message, replacement_df, data_df, data_lists

def get_lexical_data():
    norms = pd.read_csv(normspath, encoding="unicode-escape")
    similarity_matrix = np.loadtxt(similaritypath, delimiter=' ')
    frequency_list = np.array(pd.read_csv(
        frequencypath, header=None, encoding="unicode-escape")[1])
    phon_matrix = np.loadtxt(phonpath, delimiter=',')
    labels = pd.read_csv(frequencypath, header=None)[0].values.tolist()
    return norms, similarity_matrix, phon_matrix, frequency_list, labels


def calculate_model(model, history_vars, switch_names, switch_vecs):
    """
    1. Check if specified model is valid
    2. Return a set of model functions to pass
    """
    model_name = []
    model_results = []
    if model not in models:
        ex_str = "Specified model is invalid. Model must be one of the following: {models}".format(
            models=models)
        raise Exception(ex_str)
    if model == models[0] or model == models[4]:
        r1 = np.random.rand()
        r2 = np.random.rand()

        v = fmin(forage.model_static, [r1, r2], args=(
            history_vars[2], history_vars[3], history_vars[0], history_vars[1]), disp=False)

        beta_df = float(v[0])  # Optimized weight for frequency cue
        beta_ds = float(v[1])  # Optimized weight for similarity cue

        nll, nll_vec = forage.model_static_report(
            [beta_df, beta_ds], history_vars[2], history_vars[3], history_vars[0], history_vars[1])
        model_name.append('forage_static')
        model_results.append((beta_df, beta_ds, nll, nll_vec))
    if model == models[1] or model == models[4]:
        for i, switch_vec in enumerate(switch_vecs):
            r1 = np.random.rand()
            r2 = np.random.rand()

            v = fmin(forage.model_dynamic, [r1, r2], args=(
                history_vars[2], history_vars[3], history_vars[0], history_vars[1], switch_vec), disp=False)

            beta_df = float(v[0])  # Optimized weight for frequency cue
            beta_ds = float(v[1])  # Optimized weight for similarity cue

            nll, nll_vec = forage.model_dynamic_report(
                [beta_df, beta_ds], history_vars[2], history_vars[3], history_vars[0], history_vars[1], switch_vec)
            model_name.append('forage_dynamic_' + switch_names[i])
            model_results.append((beta_df, beta_ds, nll, nll_vec))
    if model == models[2] or model == models[4]:
        r1 = np.random.rand()
        r2 = np.random.rand()
        r3 = np.random.rand()
        v = fmin(forage.model_static_phon, [r1, r2, r3], args=(
            history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4], history_vars[5]), disp=False)

        beta_df = float(v[0])  # Optimized weight for frequency cue
        beta_ds = float(v[1])  # Optimized weight for similarity cue
        beta_dp = float(v[2])  # Optimized weight for phonological cue

        nll, nll_vec = forage.model_static_phon_report(
            [beta_df, beta_ds, beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4], history_vars[5])
        model_name.append('forage_phonologicalstatic')
        model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))
    if model == models[3] or model == models[4]:
        for i, switch_vec in enumerate(switch_vecs):
            # Global Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = fmin(forage.model_dynamic_phon, [r1, r2, r3], args=(
                history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4], history_vars[5], switch_vec, 'global'), disp=False)

            beta_df = float(v[0])  # Optimized weight for frequency cue
            beta_ds = float(v[1])  # Optimized weight for similarity cue
            beta_dp = float(v[2])  # Optimized weight for phonological cue

            nll, nll_vec = forage.model_dynamic_phon_report(
                [beta_df, beta_ds, beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4], history_vars[5], switch_vec, 'global')
            model_name.append(
                'forage_phonologicaldynamicglobal_' + switch_names[i])
            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))

            # Local Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = fmin(forage.model_dynamic_phon, [r1, r2, r3], args=(
                history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4], history_vars[5], switch_vec, 'local'), disp=False)

            beta_df = float(v[0])  # Optimized weight for frequency cue
            beta_ds = float(v[1])  # Optimized weight for similarity cue
            beta_dp = float(v[2])  # Optimized weight for phonological cue

            nll, nll_vec = forage.model_dynamic_phon_report(
                [beta_df, beta_ds, beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4], history_vars[5], switch_vec, 'local')
            model_name.append(
                'forage_phonologicaldynamiclocal_' + switch_names[i])
            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))

            # Switch Dynamic Phonological Model
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            v = fmin(forage.model_dynamic_phon, [r1, r2, r3], args=(
                history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4], history_vars[5], switch_vec, 'switch'), disp=False)

            beta_df = float(v[0])  # Optimized weight for frequency cue
            beta_ds = float(v[1])  # Optimized weight for similarity cue
            beta_dp = float(v[2])  # Optimized weight for phonological cue

            nll, nll_vec = forage.model_dynamic_phon_report(
                [beta_df, beta_ds, beta_dp], history_vars[2], history_vars[3], history_vars[0], history_vars[1], history_vars[4], history_vars[5], switch_vec, 'switch')
            model_name.append(
                'forage_phonologicaldynamicswitch_' + switch_names[i])

            model_results.append((beta_df, beta_ds, beta_dp, nll, nll_vec))

    # Unoptimized Model
    model_name.append('forage_random_baseline')
    nll_baseline, nll_baseline_vec = forage.model_static_report(
        beta=[0, 0], freql=history_vars[2], freqh=history_vars[3], siml=history_vars[0], simh=history_vars[1])
    model_results.append((0, 0, nll_baseline, nll_baseline_vec))
    return model_name, model_results


def calculate_switch(switch, fluency_list, semantic_similarity, phon_similarity, norms, alpha=np.arange(0, 1.1, 0.1), rise=np.arange(0, 1.25, 0.25), fall=np.arange(0, 1.25, 0.25)):
    '''
    1. Check if specified switch model is valid
    2. Return set of switches, including parameter value, if required
    '''
    switch_names = []
    switch_vecs = []

    if switch not in switch_methods:
        ex_str = "Specified switch method is invalid. Switch method must be one of the following: {switch}".format(
            switch=switch_methods)
        raise Exception(ex_str)

    if switch == switch_methods[0] or switch == switch_methods[4]:
        switch_names.append(switch_methods[0])
        switch_vecs.append(switch_simdrop(fluency_list, semantic_similarity))

    if switch == switch_methods[1] or switch == switch_methods[4]:
        for i, a in enumerate(alpha):
            switch_names.append('multimodal_alpha={alpha}'.format(alpha=a))
            switch_vecs.append(switch_multimodal(
                fluency_list, semantic_similarity, phon_similarity, a))

    if switch == switch_methods[2] or switch == switch_methods[4]:
        switch_names.append(switch_methods[2])
        switch_vecs.append(switch_troyer(fluency_list, norms))

    if switch == switch_methods[3] or switch == switch_methods[4]:
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


def run_all(data, model, switch):
    """
    Runs data, model, amd switch computations.  
    Outputs:
    - 3 dataframes for switch, model, and nll results
    """

    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
    outputs = []

    # Run through each fluency list in dataset
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        # Get History Variables
        history_vars = create_history_variables(
            fl_list, labels, similarity_matrix, frequency_list, phon_matrix)

        # Calculate Switch Vector(s)
        switch_names, switch_vecs = calculate_switch(
            switch, fl_list, history_vars[0],   history_vars[4], norms)

        # Execute Individual Model(s) and get result(s)
        model_names, model_results = calculate_model(
            model, history_vars, switch_names, switch_vecs)

        outputs.append([subj, fl_list, model_names, model_results,
                       switch_names, switch_vecs, history_vars])

    return synthesize_all_results(outputs)

def run_sims(data):

    """
    Perform only switch computations.   
    Outputs a dataframe for switch results. 
    """
    print("Running sims")
    # return synthesize_switch_results(outputs)

    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
    outputs = []

    # Run through each fluency list in dataset
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        # Get History Variables
        history_vars = create_history_variables(
            fl_list, labels, similarity_matrix, frequency_list, phon_matrix)
        
        outputs.append([subj, fl_list, history_vars])

    return synthesize_sim_results(outputs)

def run_switch(data, switch):
    """
    Perform only switch computations.   
    Outputs a dataframe for switch results. 
    """
    # return synthesize_switch_results(outputs)

    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
    outputs = []

    # Run through each fluency list in dataset
    for i, (subj, fl_list) in enumerate(tqdm(data)):
        # Get History Variables
        history_vars = create_history_variables(
            fl_list, labels, similarity_matrix, frequency_list, phon_matrix)

        # Calculate Switch Vector(s)
        switch_names, switch_vecs = calculate_switch(
            switch, fl_list, history_vars[0],   history_vars[4], norms)

        outputs.append([subj, fl_list, switch_names, switch_vecs, history_vars])

    return synthesize_switch_results(outputs)


def run_model_nll(data, model, switch):
    """
    Perform the switch computations necessary for both model and nll results.
    Outputs two dataframes containing results.
    """

    # Get Lexical Data needed for executing methods
    norms, similarity_matrix, phon_matrix, frequency_list, labels = get_lexical_data()
    outputs = []
    # Run through each fluency list in dataset
    for i, (subj, fl_list) in enumerate(tqdm(data)):

        # Get History Variables
        history_vars = create_history_variables(
            fl_list, labels, similarity_matrix, frequency_list, phon_matrix)

        # Calculate Switch Vector(s)
        switch_names, switch_vecs = calculate_switch(
            switch, fl_list, history_vars[0],   history_vars[4], norms)

        # Execute Individual Model(s) and get result(s)
        model_names, model_results = calculate_model(
            model, history_vars, switch_names, switch_vecs)

        outputs.append([subj, fl_list, model_names,
                       model_results, history_vars])

    return synthesize_model_results(outputs), synthesize_nll_results(outputs)

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
        df['Frequency_Value'] = output[2][2]
        df['Phonological_Similarity'] = output[2][4]

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
            df['Frequency_Value'] = output[4][2]
            df['Phonological_Similarity'] = output[4][4]
            switch_df.append(df)
        
        switch_df = pd.concat(switch_df, ignore_index=True)
        switch_results.append(switch_df)

    switch_results = pd.concat(switch_results, ignore_index=True)

    return switch_results


def synthesize_model_results(outputs):
    """
    Returns dataframe containing results of model computations. 
    Params:
    - Array of [subject, fluency list, model names, model results, history variables].
    - Fluency list and history vars are included as models uses same output array as nll. 
    Output:
    - model results
    """
    model_results = []

    for output in outputs:
        subj = output[0]
        model_names = output[2]
        results = output[3]

        # Create Model Output Results DataFrame
        for i, model in enumerate(model_names):
            # print(model)
            model_dict = dict()
            model_dict['Subject'] = subj
            model_dict['Model'] = model
            model_dict['Beta_Frequency'] = results[i][0]
            model_dict['Beta_Semantic'] = results[i][1]
            # print(results[i])
            # sys.exit()
            if len(results[i]) == 4:
                model_dict['Beta_Phonological'] = None
                model_dict['Negative_Log_Likelihood_Optimized'] = results[i][2]
            if len(results[i]) == 5:
                model_dict['Beta_Phonological'] = results[i][2]
                model_dict['Negative_Log_Likelihood_Optimized'] = results[i][3]
            model_results.append(model_dict)

    model_results = pd.DataFrame(model_results)

    return model_results


def synthesize_nll_results(outputs):
    """
    Returns dataframe containing results of nll computations. 
    Params:
    - Array of [subject, fluency list, model names, model results, history variables] 
    Output:
    - nll results
    """
    nll_results = []
    for output in outputs:
        subj = output[0]
        fl_list = output[1]
        model_names = output[2]
        results = output[3]

        # Create Negative Log Likelihood DataFrame with Item Wise NLL
        nll_df = pd.DataFrame()
        nll_df['Subject'] = len(fl_list) * [subj]
        nll_df['Fluency_Item'] = fl_list
        for k, result in enumerate(results):
            if len(result) == 4:
                nll_df['NLL_{model}'.format(model=model_names[k])] = result[3]
            if len(result) == 5:
                nll_df['NLL_{model}'.format(model=model_names[k])] = result[4]
        # Add freq, semantic sim, and phon sim values to itemwise nll data
        nll_df['Semantic_Similarity'] = output[4][0]
        nll_df['Frequency_Value'] = output[4][2]
        nll_df['Phonological_Similarity'] = output[4][4]
        nll_results.append(nll_df)

    nll_results = pd.concat(nll_results, ignore_index=True)

    return nll_results


def synthesize_all_results(outputs):
    """
    Returns 3 dataframes containing results of switch, model, and nll computations. 
    Output file(s):
    - All model result(s)
    - Switch Vector Result(s)
    - Item-Wise Negative Log Likelihood
    """
    model_results = []
    switch_results = []
    nll_results = []
    for output in outputs:
        subj = output[0]
        fl_list = output[1]
        model_names = output[2]
        results = output[3]
        switch_methods = output[4]
        switch_vectors = output[5]
        # Create Model Output Results DataFrame
        for i, model in enumerate(model_names):
            # print(model)
            model_dict = dict()
            model_dict['Subject'] = subj
            model_dict['Model'] = model
            model_dict['Beta_Frequency'] = results[i][0]
            model_dict['Beta_Semantic'] = results[i][1]
            # print(results[i])
            # sys.exit()
            if len(results[i]) == 4:
                model_dict['Beta_Phonological'] = None
                model_dict['Negative_Log_Likelihood_Optimized'] = results[i][2]
            if len(results[i]) == 5:
                model_dict['Beta_Phonological'] = results[i][2]
                model_dict['Negative_Log_Likelihood_Optimized'] = results[i][3]
            model_results.append(model_dict)

        # Create  Switch Results DataFrame
        switch_df = []
        for j, switch in enumerate(switch_vectors):
            df = pd.DataFrame()
            df['Subject'] = len(switch) * [subj]
            df['Fluency_Item'] = fl_list
            df['Switch_Value'] = switch
            df['Switch_Method'] = switch_methods[j]
            switch_df.append(df)

        switch_df = pd.concat(switch_df, ignore_index=True)
        switch_results.append(switch_df)

        # Create Negative Log Likelihood DataFrame with Item Wise NLL
        nll_df = pd.DataFrame()
        nll_df['Subject'] = len(fl_list) * [subj]
        nll_df['Fluency_Item'] = fl_list
        for k, result in enumerate(results):
            if len(result) == 4:
                nll_df['NLL_{model}'.format(model=model_names[k])] = result[3]
            if len(result) == 5:
                nll_df['NLL_{model}'.format(model=model_names[k])] = result[4]
        # Add freq, semantic sim, and phon sim values to itemwise nll data
        nll_df['Semantic_Similarity'] = output[6][0]
        nll_df['Frequency_Value'] = output[6][2]
        nll_df['Phonological_Similarity'] = output[6][4]
        nll_results.append(nll_df)

    model_results = pd.DataFrame(model_results)
    switch_results = pd.concat(switch_results, ignore_index=True)
    nll_results = pd.concat(nll_results, ignore_index=True)

    return model_results, switch_results, nll_results
