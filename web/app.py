from flask import Flask, render_template, request
import joblib
import os
from PIL import Image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = joblib.load('svm_model.pkl')

# Preprocessing seperti saat training
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((64, 60))                 # ukuran harus sama
    img_array = np.array(img) / 255.0          # normalisasi 0–1
    return img_array.flatten().reshape(1, -1)  # jadi shape (1, 3840)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filename = file.filename

            # Preprocess dan prediksi
            image = preprocess_image(filepath)
            prediction = model.predict(image)[0]

    label = "Memakai kacamata" if prediction == 1 else "Tidak memakai kacamata"
    return render_template('index.html', prediction=label, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
