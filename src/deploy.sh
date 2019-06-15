#!/usr/bin/env bash



AWS_PROFILE=acme-development aws cloudformation deploy --template "/"$(pwd)"/template.yaml" --stack-name sagemaker-sentiment-analysis --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM AWS::IAM::Role
