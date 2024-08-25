from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch

app = Flask(__name__)
###### This part needs to be modified to get user uploaded image #######
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
######                                                           #######

## This is a folder to store result image with labels and bounding boxes
app.config['DETECTION_FOLDER'] = 'static/detections'
os.makedirs(app.config['DETECTION_FOLDER'], exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/best.pt')  # Or use your trained model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform object detection
        results = model(filepath)
        print(results)
        results.render()  # Updates results.imgs with boxes and labels
        output_image_path = app.config['DETECTION_FOLDER']
        results.save(save_dir=output_image_path)  # Save the image with detections

        return None


if __name__ == "__main__":
    app.run(debug=True)
