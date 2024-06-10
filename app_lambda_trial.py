from flask import Flask, request, jsonify
import numpy as np
import base64
import cv2
import boto3
import json

app = Flask(__name__)

# Initialize AWS Lambda client
lambda_client = boto3.client('lambda', region_name='us-east-1')  # Update the region as necessary

def invoke_lambda(function_name, payload):
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    response_payload = json.loads(response['Payload'].read())
    return response_payload

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data.get('imageData')
    image_height = data.get('imageHeight')
    image_width = data.get('imageWidth')
    image_process = data.get('imageProcess')
    
    if image_process != 'grayscale':
        return jsonify({'status': 'error', 'message': 'Unsupported image process'})

    # Process image data
    pixel_data = np.array(list(image_data.values()))
    red_array = pixel_data[2::4]
    green_array = pixel_data[1::4]
    blue_array = pixel_data[0::4]
    rgb_image_array = np.stack((red_array, green_array, blue_array), axis=-1).reshape(image_height, image_width, 3).astype(np.uint8)

    payload = {
        'image': base64.b64encode(rgb_image_array).decode('utf-8')
    }

    function_name = 'lambda_function_grayscale'  # Update with your Lambda function name
    response_payload = invoke_lambda(function_name, payload)

    # Decode the response image
    image_data_array_edited = base64.b64decode(response_payload['image'])
    image_data_array_edited = np.frombuffer(image_data_array_edited, dtype=np.uint8)
    image_data_array_edited = image_data_array_edited.reshape((image_height, image_width))

    # Convert the NumPy array to a base64-encoded string
    _, buffer = cv2.imencode('.png', image_data_array_edited)
    image_data = base64.b64encode(buffer).decode('utf-8')

    response = {
        'status': 'success',
        'img': image_data
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
