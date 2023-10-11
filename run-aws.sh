#!/usr/bin/bash

pipenv install tensorflow

pipenv run gunicorn -w 4 --bind 0.0.0.0:8080 --access-logfile=- --timeout=1800 application:application