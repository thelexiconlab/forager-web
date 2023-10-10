#!/bin/bash
#
# This is for local testing of the sample application.
#

pipenv install

# Use gunicorn to run the web service
pipenv run gunicorn -w 4 --bind 0.0.0.0:8080 --timeout 1800 --access-logfile=- --reload application:application
