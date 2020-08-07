import requests
import os
from io import BytesIO
from flask import Flask, render_template, request, jsonify
import PIL
import numpy as np
# from waitress import serve

from src.inference import Inference

app = Flask(__name__)
inf = Inference()

STYLE_CHOICE = 'https://i.pinimg.com/736x/2e/3d/8c/2e3d8c63ee274765543a01f4d495e1c5.jpg'
SAVE_LOC = os.getcwd() + '/src/static/assets/stylized' +\
    str(np.random.random()) + '.png'

response = BytesIO(requests.get(STYLE_CHOICE).content)

@app.route('/', methods=['GET'])
def status():
    return render_template('index.html')


@app.route('/model', methods=['GET'])
def train():
    return render_template('model.html')


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    results = {'status': 'fail', 'loc': SAVE_LOC}
    if request.method == 'POST':
        # Content Image
        im = request.files['file'].read()
        content_img = inf.load_img(im)
        content_img = inf.preprocess_content_image(content_img)

        # Style Image
        style_choice = response  # TYPE IO BYTES
        style_img = inf.load_style_img(style_choice)
        style_img = inf.preprocess_style_image(style_img)

        style_bottleneck = inf.run_style_predict(style_img)
        stylized_img = inf.run_style_transform(style_bottleneck, content_img)

        stylized = tensor_to_image(stylized_img)
        stylized.save(SAVE_LOC)
        results['status'] = 'pass'

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
    # For production mode, comment the line above and uncomment below
    # serve(app, host="0.0.0.0", port=8000)
