from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)




@app.route('/load_model')
def load_model_route():
    global model
    # Load the pre-trained model
    model = load_model('C:/Users/Santhosh/Documents/Python/models/model_1.h5')
    
    return 'Model loaded successfully!'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return 'No file uploaded'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No file selected'

        img = Image.open(file)
        img = img.resize((224, 224))  
        img = np.array(img) / 255.0         
        
        result = model.predict(np.expand_dims(img, axis=0))
        
        return 'Inference result: ' + str(result)
    
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
