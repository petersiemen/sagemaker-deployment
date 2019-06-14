#!/usr/bin/env bash


AWS_PROFILE=acme-development aws cloudformation validate-template --template-body "file://"$(pwd)"/template.yaml"
