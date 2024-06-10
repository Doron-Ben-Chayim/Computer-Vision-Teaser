import json
import base64
import numpy as np
import cv2

def lambda_handler(event, context):
    image_data = base64.b64decode(event['image'])
    image_data = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    _, buffer = cv2.imencode('.png', gray_image)
    gray_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        'statusCode': 200,
        'body': json.dumps({'image': gray_image_base64})
    }
