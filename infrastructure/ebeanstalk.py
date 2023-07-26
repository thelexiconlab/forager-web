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
    aws_iam as iam,
    aws_elasticbeanstalk as elasticbeanstalk,
    App, Stack
)

from constructs import Construct

class ForagerBeanStack(Stack):
    def __init__(self, scope:Construct, id:str,
            vpc_name = None,
            **kwargs) -> None:
        super().__init__(app, id, **kwargs)

        # VPC with public subnet for database
        vpc = ec2.Vpc.from_lookup(self, "vpc",
            # is_default=True
            # vpc_id = "vpc-01b8470d4eb93b108"
            vpc_name = vpc_name
        )

        # service role
        application = elasticbeanstalk.CfnApplication(self, f'{id}-app',
            application_name=id)

        version = elasticbeanstalk.CfnApplicationVersion(self, f'{id}-version',
            application_name=id,
            source_bundle=elasticbeanstalk.CfnApplicationVersion.SourceBundleProperty(
                s3_bucket="s3Bucket",
                s3_key="s3Key"
            ),

        options = [
            elasticbeanstalk.CfnEnvironment.OptionSettingProperty(
                namespace="aws:autoscaling:launchconfiguration",
                option_name="InstanceType",
                value="t3.small"
            ),
            elasticbeanstalk.CfnEnvironment.OptionSettingProperty(
                namespace="aws:autoscaling:launchconfiguration",
                option_name="IamInstanceProfile",
                value='aws-elasticbeanstalk-ec2-role'
            ),
            elasticbeanstalk.CfnEnvironment.OptionSettingProperty(
                namespace="aws:elasticbeanstalk:container:nodejs",
                option_name="NodeVersion",
                value='10.16.3'
            )
        ]

        environment = elasticbeanstalk.CfnEnvironment(self, f'{id}-env',
            environment_name=f'{id}-env',
            application_name=application.name,
            solution_stack_name="solutionStackName",
            option_settings=options,
            version_label=version.ref
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

# Bowdoin Cloud Tags
# https://bowdoin.atlassian.net/wiki/spaces/ITKB/pages/2164589/Cloud+Tagging+Standards
Tags.of(app).add('environment', get_context('deploymentType'))
for tag, value in get_context('tags').items():
    Tags.of(app).add(tag, value)

Tags.of(app).add('project', get_context('project'))

ForagerBeanStack(app, get_context('name'),
    env={ "region": get_context('region'), "account": get_context('account') }, 
    description=get_context('description'),
    vpc_name=get_context('vpc_name')
    )

app.synth()