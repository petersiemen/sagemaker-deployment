#!/bin/bash

set -ueo pipefail

sam deploy --template-file packaged.yaml --stack-name sentiment-analysis --capabilities CAPABILITY_IAM
aws cloudformation describe-stacks  --stack-name sentiment-analysis
