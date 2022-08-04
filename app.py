# Import libraries
import tensorflow as tf
import numpy as np
import os

from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras_preprocessing.image import img_to_array, load_img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('./model/model_batik_10_kelas.h5')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    class_dict = {'Batik Cendrawasih': 0, 'Batik Dayak': 1, 'Batik Ikat Celup': 2, 'Batik Insang': 3, 'Batik Kawung': 4,
                  'Batik Megamendung': 5, 'Batik Parang': 6, 'Batik Poleng': 7, 'Batik Sekar Jagad': 8,
                  'Batik Tambal': 9}

    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            image_size = os.stat(image_path).st_size
            image_format = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']

            if image.filename.split('.')[-1] in image_format:
                if image_size <= 1024000:
                    img = load_img(image_path, target_size=(224, 224))
                    img_array = img_to_array(img) / 255.0
                    img_array = tf.expand_dims(img_array, 0)

                    motives_list = list(class_dict.keys())
                    prediction = model.predict(img_array)
                    pred_idx = np.argmax(prediction)
                    pred_motive = motives_list[pred_idx]
                    pred_confidence = prediction[0][pred_idx] * 100

                    classification = (pred_motive, round(pred_confidence, 2))

                    return render_template('index.html', uploaded_image=image.filename, data=classification)
                else:
                    message = ['Ukuran gambar tidak boleh lebih dari 1 Mb!']

                    if os.path.isfile(image_path):
                        os.remove(image_path)

                    return render_template('index.html', warning=message)
            else:
                if os.path.isfile(image_path):
                    os.remove(image_path)
                message = ['Format gambar harus jpg, png, dan jpeg']
                return render_template('index.html', warning=message)


@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
