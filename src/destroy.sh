#!/usr/bin/env bash



AWS_PROFILE=acme-development aws cloudformation delete-stack --stack-name sagemaker-sentiment-analysis
