#!/usr/bin/env python3

# Forager Web Infrastructure AWS
# 
# The following are the components of the supporting infrastructure
# - VPC and 2 public subnets for application
#
# Not part of this infrastructure, managed manually
# - Database schema creation; application user and tables
# - AppRunner for deploying Python Flask / React Application
#
# 2022-03-13 Stephen Houser <houser@bowdoin.edu>
from aws_cdk import (
    Tags,
    aws_ec2 as ec2,
    aws_apprunner_alpha as apprunner,
    App, Stack
)

from constructs import Construct

BUILD_COMMAND='pip install pipenv && pipenv install'
START_COMMAND='pipenv run gunicorn -w 4 --bind 0.0.0.0:8080 --access-logfile=- --timeout=1800 application:application'

class SimpleDBStack(Stack):
    def __init__(self, scope:Construct, id:str,
            instance_type = None,       ## ec2.InstanceType
            github_connection_arn = None,
            github_repo = None,
            github_branch = None,   
            vpc_name = None,      
            **kwargs) -> None:
        super().__init__(app, id, **kwargs)

        # VPC with public subnet for database 
        vpc = ec2.Vpc.from_lookup(self, "vpc",
            # is_default=True
            # vpc_id = "vpc-01b8470d4eb93b108"
            vpc_name = vpc_name        
        )

        if not instance_type:
            instance_type = ec2.InstanceType.of(
                ec2.InstanceClass.BURSTABLE3, ec2.InstanceSize.MICRO)

        ##
        ## AppRunner Service
        ##
        vpc_connector = apprunner.VpcConnector(self, 'vpcConnector', 
            vpc = vpc,
            vpc_subnets = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC)
        )

        service = apprunner.Service(self, id,
            service_name = id,
            vpc_connector = vpc_connector,
            source = apprunner.Source.from_git_hub(
                connection = apprunner.GitHubConnection.from_connection_arn(github_connection_arn),
                repository_url = github_repo,
                branch = github_branch,
                # Use this to configue via apprunner.yaml
                # configuration_source = apprunner.ConfigurationSourceType.REPOSITORY,
                # Use this (2 lines) to configure manually
                configuration_source = apprunner.ConfigurationSourceType.API,
                code_configuration_values=apprunner.CodeConfigurationValues(
                    runtime=apprunner.Runtime.PYTHON_3,
                    start_command=START_COMMAND,
                    build_command=BUILD_COMMAND,
                )
            )
        )
  
#
# Instantiate the app and stack
#
def get_context(key):
    deployment_type = app.node.try_get_context('deploymentType')
    if not deployment_type:
        deployment_type = 'dev'

    deployment_config = app.node.try_get_context(deployment_type)

    # use deploymentType value if present
    if key in deployment_config:
        return deployment_config[key]
    
    # otherwise use default value in general context
    return app.node.try_get_context(key)

app = App()
Tags.of(app).add('project', get_context('project'))

SimpleDBStack(app, get_context('name'), 
    env={ "region": get_context('region'), "account": get_context('account') }, 
    description=get_context('description'),
    vpc_name=get_context('vpc_name'),
    github_connection_arn=get_context('github_connection_arn'),
    github_repo=get_context('github_repo'),
    github_branch=get_context('github_branch')
    )

app.synth()