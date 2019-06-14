import boto3


def lambda_handler(event, context):
    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    # runtime = boto3.Session().client('sagemaker-runtime')
    #
    # # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    # response = runtime.invoke_endpoint(EndpointName='**ENDPOINT NAME HERE**',  # The name of the endpoint we created
    #                                    ContentType='text/plain',  # The data format that is expected
    #                                    Body=event['body'])  # The actual review

    # The response is an HTTP response whose body contains the result of our inference
    #result = response['Body'].read().decode('utf-8')
    result = "juchhu"

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*'},
        'body': result
    }