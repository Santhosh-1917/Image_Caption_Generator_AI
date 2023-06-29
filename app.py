from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import uuid

app = Flask(__name__)

# Load the pre-trained model
model = None

@app.route('/load_model')
def load_model_route():
    global model
    try:
        # Load the pre-trained model
        model = load_model('C:/Users/Santhosh/Documents/Python/models/model_1.h5')
        return 'Model loaded successfully!'
    except Exception as e:
        return 'Error loading the model: ' + str(e)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if files were uploaded
        if 'files' not in request.files:
            return 'No files uploaded'
        
        files = request.files.getlist('files')
        
        captions = []
        for file in files:
            if file.filename == '':
                return 'No file selected'
            
            try:
                # Save the uploaded image with a unique filename
                filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
                filepath = os.path.join('static/uploads', filename)
                file.save(filepath)
                
                # Open and preprocess the image
                img = Image.open(filepath)
                img = img.resize((224, 224))
                img_array = img_to_array(img)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Perform inference using the loaded model
                result = model.predict(img_array)
                
                # Decode the prediction and get the class label
                class_label = decode_prediction(result)
                
                captions.append({
                    'image': filepath,
                    'caption': class_label
                })
            except Exception as e:
                return 'Error during inference: ' + str(e)

        return render_template('result.html', captions=captions)

    return render_template('index.html')

def decode_prediction(result):
   \
    class_labels = ['Class 1', 'Class 2', 'Class 3']
    predicted_class_index = np.argmax(result)
    class_label = class_labels[predicted_class_index]
    return class_label

if __name__ == '__main__':
    app.run(debug=False)

