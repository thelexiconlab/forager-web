#!/bin/bash

# default to dev deployment, unless prod given on command line
type="${1:-dev}"
if [ $type = "prod" ]; then
	echo "*** Deploying to PRODUCTION ***"
elif [ $type = "dev" ]; then
	echo "*** Deploying to development ***"
else
	echo "\"$type\" is not a valid deployment type"
	exit 1
fi

(cd .. & pipenv install -d)

pipenv run cdk --profile faculty --context deploymentType=${type} deploy

echo "Infrastructure deployed!"
echo "To delete:"
echo "    pipenv run cdk --profile faculty destroy"
echo ""
echo "To show CDK generated code"
echo "	  pipenv run cdk --profile faculty synth > cdk.yaml"