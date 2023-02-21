# Deployment Guide to AWS Elastic Beanstalk

Short deployment notes for the Forager application to AWS Elastic Beanstalk.
This is using the console rather than the AWS CLI or CDK. Just for simplicity
and speed in getting a proof-of-concept up and running.

## Deploy a New Version

1. Bundle the `zip` file with the most recent code. Make sure to increment the version number. Otherwise beanstalk won't update it.

```
VERSION=v0.1
cd forager
find . -name __pycache__ -exec rm -rf {} \;
zip ../forager-${VERSION}.zip -r *
cd ..
```

2. Open up the AWS Console and go to Elastic Beanstalk, Choose the Environment (ForagerKumar-dev), Choose "Upload and Deploy"

3. Surf. Current URL is: http://foragerkumar-dev.eba-vqujhnts.us-east-1.elasticbeanstalk.com/ (2023/01/19). This is also available in the console.

## Initial Environment Creation

AWS Elastic Beanstalk hosts an _Application_ that can contain several
_Environments_. The environments can have different versions of the code,
different options, and individual URLs.

1. Create an _Application_
    - Application Name: Forager-Kumar
    - Application Tags: project : forager-kumar

2. Create an _Environment_:
    - Web server environment
    - Environment name: forager-dev (or prod)
    - Platform: Managed, Python
    - Platform Branch and version: choose the latest
    - Source code origin Version Label: v0.1, upload file
    - Application Code Tags: project : forager-kumar
    - *configure more options*
    - Network: 
        - VPC: `faculty-projects`
        - Public IP Address
        - Select the two public ip subnets
        - Save
    - Tags: project : forager-kumar
    - Create Environment



