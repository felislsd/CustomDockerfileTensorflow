
#!/usr/bin/env python3
# Import required libraries
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import csv
import os
import uuid
import imghdr  # for file content validation

# Initialize Flask application
app = Flask(__name__)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = '/mnt/b/BreadLoafCats/BreadLoadAppData'
RESULT_FOLDER = '/mnt/b/BreadLoafCats/BreadLoadAppData'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER




def file_check(filename):
    if imghdr.what(filename) in ALLOWED_EXTENSIONS:
        return True
    return False
    


def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        
            
            # save the file to the upload folder
        if file and file_check(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # process the file (classify bread loaf cat or not)
            # (this part will be added once we have the classification logic)
            
            # return a response (e.g., showing the result)
            return 'File uploaded successfully'



# Create CSV file to store user information
CSV_FILE = 'user_info.csv'

# Load pretrained model
#model_path = '/mnt/b/BreadLoafCats/scripts/breadloaf_cat_model_v3.h5'
model_path = '/scripts/breadloaf_cat_model_v3.h5'
model = load_model(model_path)

# Define image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Define prediction function
def predict_image(image_path, model):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction[0][0]

# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        # Check if the file is an allowed format
        if file:
            # Get user information
            user_name = request.form.get('username')
            
            filename = str(uuid.uuid4()) + '_' + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            prediction = predict_image(filepath, model)
            
            # Calculate prediction probability
            probability = prediction * 100
            
            # Display prediction result
            result = f'Probability: {probability:.2f}%<br>'
            if prediction > 0.5:
                result += 'Breadloaf Cat'
            else:
                result += 'Not a Breadloaf Cat'
            
            # Save image with prediction result
            result_filepath = os.path.join(app.config['RESULT_FOLDER'], 'result_' + filename)
            os.rename(filepath, result_filepath)
            
            # Save user information, image filename, and prediction result to CSV file
            with open(CSV_FILE, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([user_name, filename, result])
            
            return render_template('index.html', message=result, image=filename)
    
    return render_template('index.html')

# Serve uploaded images
@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file_route():
    return upload_file()

# Run the application
if __name__ == '__main__':
    if not os.path.exists(app.config['RESULT_FOLDER']):
        os.makedirs(app.config['RESULT_FOLDER'])
    
    # Check if CSV file exists, if not create it with headers
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Username', 'Image Filename', 'Prediction Result'])
    
    app.run(debug=True)
