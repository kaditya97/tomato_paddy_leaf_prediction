from flask import Flask, render_template, request
import keras
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__, static_url_path='')


@app.route("/", methods=["GET","POST"])
def index():
    np.set_printoptions(suppress=True)
    model = keras.models.load_model('vgg16.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    if request.method == 'POST':
        predict = ''
        val = 0
        img = request.files['images']
        image = Image.open(img)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        if(prediction[0][0] > 0.5):
            val = prediction[0][0]
            predict = "Other"
        elif(prediction[0][1] > 0.5):
            val = prediction[0][1]
            predict = "Paddy_leaf"
        elif(prediction[0][2] > 0.5):
            val = prediction[0][2]
            predict = "Tomato_leaf"
        else:
            predict = "Please upload a clear image"
        return render_template('index.html', prediction_text = predict, val = val)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
