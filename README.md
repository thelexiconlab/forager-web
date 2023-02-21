# lexicon-web

# Forager 

A web application which accepts uploaded .txt data. The website runs the forager module with the selected switch and model, returning the results as a zip files containing user csvs. 

## Dependencies
- Python (dependencies in `requirements.txt`)
    - Flask
    - Scipy
    - Pandas
    - nltk
- angularJS 

## How to run webpage 
- Run api_router.py. The terminal should indicate that the local server is active. 
- Open 'http://localhost:5000' on any browser to display the homepage. 

## Changes from forager package
- run_foraging.py is replaced by run_foraging_web.py. 
- added methods to utils.py 

