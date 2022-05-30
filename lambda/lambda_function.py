import json
import boto3
from botocore.exceptions import ClientError
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences
import email
import os

ENDPOINT_NAME = "sms-spam-classifier-mxnet-2022-04-05-17-22-29-352"

def lambda_handler(event, context):
    # get the email from S3
    s3 = boto3.client('s3', region_name='us-east-1')
    info = event['Records'][0]['s3']
    bucket_name = info['bucket']['name']
    obejct_key = info['object']['key']
    file = s3.get_object(Bucket=bucket_name, Key=obejct_key)['Body'].read()
    message = email.message_from_string(file.decode("utf-8"))
    body = get_email_body(message)
    print("---------body is-------:", body)
 
    # Sagemaker
    response = get_prediction(body)
    # get the score, classfication and probability of the prediction
    score = int(response["predicted_label"][0][0])
    classification = 'HAM' if score == 0 else "SPAM"
    probability = response["predicted_probability"][0][0]
    
    # send the reply email
    sample_body = get_sample_body(body)
    send_email(message, sample_body, classification, probability)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Successfully predict whether a message is spam or not')
    }

def get_prediction(body):
    vocabulary_length = 9013
    test_msg = [body]
    one_hot_test_messages = one_hot_encode(test_msg, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    encoded_json_msg = json.dumps(encoded_test_messages.tolist())
    
    sagemaker = boto3.client('runtime.sagemaker', region_name='us-east-1')
    response = sagemaker.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                        ContentType='application/json',
                                        Body=encoded_json_msg)
    print(response['Body'])
    response = json.loads(response['Body'].read().decode())
    return response
    
    
def get_email_body(message):
    body = ''
    for part in message.walk():
        if part.get_content_type() == 'text/plain':
            body = part.get_payload()
    body = body.rstrip()
    return body
    
def get_sample_body(body):
    if len(body) > 240:
        sample_body = body[:240]
    else:
        sample_body = body
    return sample_body
    
# function to send email by using SES
def send_email(message, sample_body, classification, probability):
    # prepare the email context to send
    SENDER = message['From']
    # RECIPIENT = message['To']
    RECIPIENT ="zc2569@columbia.edu"
    SUBJECT = "Your email report is here!"
    CHARSET = "UTF-8"
    receive_date = message['Date']
    receive_date = receive_date[:-5]
    print("-------date is------", receive_date)
    email_subject = message['SUBJECT']
    email_content = "We received your email sent at {} with the subject {}.\r\n".format(receive_date, email_subject)
    email_content += "\r\nHere is a 240 character sample of the email body:\r\n"
    email_content += "\r\n" + sample_body + "\r\n"
    email_content += "\r\nThe email was categorized as {} ".format(classification)
    email_content += "with a {:.5%} confidence.".format(probability)
    

    # Try to send the email.
    client = boto3.client('ses',region_name = 'us-east-1')
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination = {
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message = {
                'Body': {
                    'Text': {
                        'Charset': CHARSET,
                        'Data': email_content,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
        )
    # display error if something went wrong
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Successfully sent the email.")
