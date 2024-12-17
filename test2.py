import boto3
import json

# Initialize the Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="eu-central-1"
)

def invoke_llama3(input_text):
    # Payload for Meta Llama 3
    payload = {
        "prompt": input_text,  # Llama 3 requires "prompt" key for input
        "max_gen_len": 512,    # Maximum generation length
        "temperature": 0.7,    # Adjust temperature for creativity
    }

    # Invoke the model with Inference Profile ARN
    response = bedrock_client.invoke_model(
        modelId="meta.llama3-2-3b-instruct-v1:0",
        inferenceProfileArn="arn:aws:bedrock:eu-central-1:011528279330:inference-profile/eu.meta.llama3-2-3b-instruct-v1:0",  # Replace with your ARN
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    # Read the response body
    response_body = response['body'].read()
    result = json.loads(response_body)
    return result

# Test the function
user_input = "What is the capital of France?"
output = invoke_llama3(user_input)
print("Llama 3 Response:", output)
