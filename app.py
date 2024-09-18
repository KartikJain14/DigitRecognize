from flask import Flask, Response, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the model
model = load_model('digit_recognizer_model.keras')

def preprocess_image(image_bytes):
    try:
        original_img = Image.open(io.BytesIO(image_bytes))
        original_img.save('original_image.png')
        img = original_img.convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        preprocessed_img = Image.fromarray((img_array.squeeze() * 255).astype(np.uint8))
        preprocessed_img.save('preprocessed_image.png')
        return img_array
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
	return favicon.ico

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files.get('image')
        if not image_file:
            return jsonify({'error': 'No image uploaded'}), 400

        img_bytes = image_file.read()
        img_array = preprocess_image(img_bytes)

        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # Send entire prediction array and predicted digit
        return jsonify({
            'digit': int(predicted_digit),
            'prediction': prediction.tolist()  # Convert NumPy array to list for JSON serialization
        })
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
