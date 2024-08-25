from flask import Flask, request, jsonify
import torch
import requests
from PIL import Image
from io import BytesIO

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt')

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_objects():
    # Get the image URL from the request
    image_url = request.json.get('image_url')
    
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400
    
    # Download the image
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    except Exception as e:
        return jsonify({'error': f'Error downloading image: {str(e)}'}), 400
    
    # Perform object detection
    results = model(img)
    
    # Extract the results
    detected_objects = results.pandas().xyxy[0].to_dict(orient="records")
    
    # Format the response to bool value base on the number of detected objects
    isDrunk = False
    if len(detected_objects) > 0:
        isDrunk = True
    else:
        isDrunk = False

    # Format the response
    response = {
        'detected_objects': isDrunk
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
