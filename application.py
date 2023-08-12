# Routes HTTP requests performed by homepage to appropriate functions and pages.


# Imports
import os, sys
from flask import Flask, render_template, request, send_file, abort,jsonify, session
from io import BytesIO
from zipfile import ZipFile
from run_foraging_web import *
import base64
import json




application = Flask(__name__)
application.config['SECRET_KEY'] = 'XYZ'

# set a global variable for the data_lists that will be used to store the data
# this will be updated once evaluate-data is called

data_lists = None

# Change variable notation for angularJS compatibility
jinja_options = application.jinja_options.copy()
jinja_options.update(dict(
    block_start_string='<%',
    block_end_string='%>',
    variable_start_string='%%',
    variable_end_string='%%',
    comment_start_string='<#',
    comment_end_string='#>',
))
application.jinja_options = jinja_options

# Initial homepage
@application.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

# Request to evaluate file data. Returns homepage with replacement/truncation results.
# needs to present a new button for people to download the replacements/truncations

@application.route('/evaluate-data', methods=['POST'])
def get_data_evaluation():
    evaluation_message = "No file uploaded."
    data_results = None
    f = request.files['filename']
    if f: 
        user_oov_choice = request.form['selected-oov']
        evaluation_message, replacement_df, data_df, data_lists = get_evaluation_message(f, user_oov_choice)
        data_results = get_data_results(evaluation_message, replacement_df, data_df)
    else:
        abort(400)
    
    if data_results is None:
        abort(400)
    else:
        data_lists = [(str(idx), words) for idx, words in data_lists]
        session['data_lists'] = json.dumps(data_lists)
        return data_results

def get_data_results(eval_msg, replacement_df, data_df):

    # Prepare data and run model for selected features
    
    results = {"evaluation_df": replacement_df,
                "data_df": data_df}
    
    # Create a BytesIO buffer to store the zip file content
    zip_stream = BytesIO()
    with ZipFile(zip_stream, 'w') as zf:
        # Write each result as a csv file
        for name, result in results.items():
            filename = name + ".csv"
            zf.writestr(filename, result.to_csv(index=False))

    # Create a base64-encoded version of the zip content
    zip_base64 = base64.b64encode(zip_stream.getvalue()).decode('utf-8')

    # Prepare the response data
    response_data = {
        'message': eval_msg,
        'zipContent': zip_base64
    }

    # Return the response as JSON
    return jsonify(response_data)

# Request to evaluate data. Returns downloadable results if 
# successful and 400 error if not.
@application.route('/run-model', methods=['POST'])
def upload_file():
    results = None

    # Process file
    f = request.files['filename']
    if f:  
        if 'selected-sims' in request.form:
            simval = request.form['selected-sims']
            results = get_results(f, simval)
        elif 'selected-switch' in request.form:
            switch = request.form['selected-switch']
            results = get_results(f, switch)
        else:
            abort(400)
    else:
        abort(400)
    
    if results is None:
        abort(400)
    else: 
        return results




# Compute results files. Returns Zipfile containing outputs, or none if error.  
def get_results(file, switch):

    # Prepare data and run model for selected features
    try:
        stored_data_lists = json.loads(session.get('data_lists', []))
        print("stored_data_lists=", stored_data_lists)

        if switch != "sims":
            switch_results = run_switch(stored_data_lists, switch)
            results = {"switch_results" : switch_results}
        else:
            sim_results = run_sims(stored_data_lists)
            results = {"sim_results" : sim_results}
    except: 
        return None

    # Compress results into zip file
    filename_head = file.filename.split(".")[0] + '_forager'
    #filename_head = file.filename.split(".")[0] + '_model_' + model + '_switch_' + switch
    zip_stream = BytesIO()
    with ZipFile(zip_stream, 'w') as zf:
        # Write each result as a csv file
        for name, result in results.items(): 
            filename = filename_head + "_" + name + ".csv"
            zf.writestr(filename, result.to_csv())
    zip_stream.seek(0)

    return send_file(
        zip_stream,
        as_attachment = True,
        download_name= filename_head + '_results.zip'
    )

# About Page
@application.route('/about', methods = ['GET'])
def about():
    return render_template('about.html')
        

# Docs Page
@application.route('/docs', methods = ['GET'])
def docs():
    return render_template('docs.html')

# cite Page
@application.route('/cite', methods = ['GET'])
def cite():
    return render_template('cite.html')

# About Page
@application.route('/contact', methods = ['GET'])
def contact():
    return render_template('contact.html')


# Run on local server
if __name__ == '__main__':
    # application.debug = True  # Refreshes server when code is changed

    # AWS Elasticbeanstalk proxies to port 8000 by default
    application.run(host='127.0.0.1', port=8000)
