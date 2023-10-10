# Deployment Guide to AWS AppRunner

Short deployment notes for the Forager application to AWS AppRunner.

## Local Development

The root directory of the project contains a `run-local.sh` script to aid in running the application in an environment as similar to AWS AppRunner as possible. This includes running it within the `gunicorn` WSGI server. To develp and run locally, in the project's root directory:

```bash
$ ./run-local.sh

Installing dependencies from Pipfile.lock (da155c)...
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
[2023-03-06 11:28:41 -0500] [68648] [INFO] Starting gunicorn 20.1.0
[2023-03-06 11:28:41 -0500] [68648] [INFO] Listening at: http://0.0.0.0:8080 (68648)
[2023-03-06 11:28:41 -0500] [68648] [INFO] Using worker: sync
[2023-03-06 11:28:41 -0500] [68649] [INFO] Booting worker with pid: 68649
[2023-03-06 11:28:41 -0500] [68651] [INFO] Booting worker with pid: 68651
[2023-03-06 11:28:41 -0500] [68652] [INFO] Booting worker with pid: 68652
[2023-03-06 11:28:41 -0500] [68653] [INFO] Booting worker with pid: 68653
...
```

The server will start and print out the local http address that can be used to view the web application.

This script should also install any needed dependencies via `pipenv`.

## How to deploy a new version

The application is configured to automatically deploy new code versions from the `main` branch from GitHub. There should be no action required to deploy beyond a `git push` to the `main` branch.

AppRunner will check the health of the deployed application and roll back to the most recent working version if something goes amiss.

## Initial Environment Creation

The initial environment is created through the AWS Cloud Development Kit (AWS CDK). The code for the CDK is in the `infrastructure.py` file with variable options set in the `cdk.json` file. A `deploy.sh` script is included to assist in remembering the command(s) to update or deploy new infrastructure.

The only difference between `infrastructure.py` and the delpoyed infrastructure is a manual change in the AWS console to *automatic deployment*. The AWS CDK has no option to configure this (as of 3/6/2023) and *manual deployment* is the default.