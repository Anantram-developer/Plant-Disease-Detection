from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import tensorflow as tf
import os  # Added to handle folder creation

app = Flask(__name__)

# --- CONFIGURATION ---
# Create the upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploadimages'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
try:
    model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load Disease Info
try:
    with open("plant_disease.json", 'r') as file:
        plant_disease = json.load(file)
    print("Disease JSON loaded successfully.")
except Exception as e:
    print(f"Error loading JSON: {e}")


@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature


def model_predict(image_path):
    img = extract_features(image_path)
    prediction = model.predict(img)

    # Get the index of the highest probability
    predicted_index = prediction.argmax()

    # Get the dictionary from your JSON list
    result_data = plant_disease[predicted_index]

    # Create a copy so we can modify the name for display without changing the original data
    response = result_data.copy()

    # Clean up the name: "Apple___Black_rot" -> "Apple - Black rot"
    response['name'] = response['name'].replace('___', ' - ').replace('_', ' ')

    return response


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        if 'img' not in request.files:
            return redirect('/')

        image = request.files['img']

        if image.filename == '':
            return redirect('/')

        unique_id = uuid.uuid4().hex
        filename = f"temp_{unique_id}_{image.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save image
        image.save(file_path)
        print(f'Saved file to: {file_path}')

        # Predict
        prediction_result = model_predict(file_path)

        return render_template('index.html',
                               result=True,
                               imagepath=f'/uploadimages/{filename}',
                               prediction=prediction_result)

    else:
        return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)
