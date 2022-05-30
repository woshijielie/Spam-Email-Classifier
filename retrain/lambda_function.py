import json
import boto3


def lambda_handler(event, context):
    # Sagemaker
    client = boto3.client('runtime.sagemaker', region_name='us-east-1')
    
    #wish to get current status of instance
    status = client.describe_notebook_instance(NotebookInstanceName='sms_spam_classifier_mxnet')

    #Start the instance
    client.start_notebook_instance(NotebookInstanceName='sms_spam_classifier_mxnet')
    
    return {
        'statusCode': 200,
        'body': json.dumps('Successfully restart the notebook instance.')
    }
