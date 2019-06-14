# sam-app

## Development
We use pipenv to develop. 

1. Install [pipenv](https://docs.pipenv.org/en/latest/)
2. Create your virtualenv using pipenv by running 
    ```bash
    pipenv install
    ```
3. Activate pipenv
   ```bash
    pipenv shell
    ```
    
    
## Build and Deploy the lambda function using [aws-sam-cli](https://github.com/awslabs/aws-sam-cli)

1. Install aws-sam-cli
2. Create a requirements.txt from the latest Pipfile
    ```bash
    pipenv lock -r > src/requirements.txt
    ```
    
3. Test locally
    1. Test AWS Lambda function locally 
    ```bash
    sam build
    sam local invoke "SentimentAnalysisFunction" -e event.json
    echo '{}' | sam local invoke "SentimentAnalysisFunction"
    ```
    2. Test API Gateway locally
   ```bash
    sam local start-api
    ```

4. Validate, Build & Package AWS Lambda function
    ```bash
    AWS_PROFILE=acme-development ./validate_build_and_package.sh
    ```
    
5. Deploy AWS Lambda function
    ```bash
    AWS_PROFILE=acme-development ./deploy.sh
    ```
