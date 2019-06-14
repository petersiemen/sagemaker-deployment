#!/bin/bash

set -ueo pipefail

S3_BUCKET="acme-development-lamdba-functions"
LAMDA_FUNCTION="sentiment-analysis"


TAG="$(git describe  --all --long | cut -d "/" -f2)"
export TAG
echo "TAG=${TAG}" > src/version


echo "Releasing ${LAMDA_FUNCTION}"
sam validate
sam build --use-container
sam package --s3-bucket "${S3_BUCKET}" --s3-prefix "sam/${LAMDA_FUNCTION}/${TAG}" --output-template-file packaged.yaml

