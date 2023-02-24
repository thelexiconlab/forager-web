#!/bin/bash

(cd .. & pipenv install -d)
# Development deploy
pipenv run cdk --profile faculty deploy

# Production deploy
# pipenv run cdk --profile faculty --context deployType=prod deploy

echo "Infrastructure deployed!"
echo "To delete:"
echo "    pipenv run cdk --profile faculty destroy"
echo ""
echo "To show CDK generated code"
echo "	  pipenv run cdk --profile faculty synth > cdk.yaml"