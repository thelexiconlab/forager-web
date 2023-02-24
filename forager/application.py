# Routes HTTP requests performed by homepage to appropriate functions and pages.


# Imports
import os, sys
from flask import Flask, render_template, request, send_file, abort
from io import BytesIO
from zipfile import ZipFile
#from run_foraging_web import *


application = Flask(__name__)

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
@application.route('/evaluate-data', methods=['POST'])
def get_data_evaluation():
    evaluation_message = "No file uploaded."
    f = request.files['filename']
    if f: 
        evaluation_message = get_evaluation_message(f)
    return evaluation_message

# Request to evaluate data. Returns downloadable results if 
# successful and 400 error if not.
@application.route('/run-model', methods=['POST'])
def upload_file():
    results = None
    
    calc_switch = True if (request.args.get('switch') == 'true') else False
    calc_model_nll = True if (request.args.get('model') == 'true') else False

    # Process file
    f = request.files['filename']
    if f:  
        model = request.form['selected-model']
        switch = request.form['selected-switch']
        results = get_results(f, model, switch, calc_switch, calc_model_nll)
    else:
        abort(400)
    
    if results == None:
        abort(400)
    else: 
        return results


# Compute results files. Returns Zipfile containing outputs, or none if error.  
def get_results(file, model, switch, calc_switch, calc_model_nll):

    # Prepare data and run model for selected features
    try:
        data = retrieve_data(file)
    
        if(calc_model_nll & calc_switch):
            model_results, switch_results, nll_results = run_all(data, model, switch)
            results = {"model_results" : model_results,
                "switch_results" : switch_results,
                "nll_results" : nll_results
            }
        elif(calc_switch):
            switch_results = run_switch(data, model, switch)
            results = {"switch_results" : switch_results}
            model = "none"
        elif(calc_model_nll):
            model_results, nll_results = run_model_nll(data, model, switch)
            results = {"model_results" : model_results,
                "nll_results" : nll_results
            }
            switch = "none"
    except: 
        return None

    # Compress results into zip file
    filename_head = file.filename.split(".")[0] + '_model_' + model + '_switch_' + switch
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


# Download Page
@application.route('/download', methods = ['GET'])
def download():
    return render_template('download.html')


# About Page
@application.route('/contact', methods = ['GET'])
def contact():
    return render_template('contact.html')


# Run on local server
if __name__ == '__main__':
    # application.debug = True  # Refreshes server when code is changed

    # AWS Elasticbeanstalk proxies to port 8000 by default
    application.run(host='127.0.0.1', port=8000)
