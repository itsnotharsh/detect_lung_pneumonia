from flask import Flask, render_template, request
from models import MyModels
import os
import io
from PIL import Image

app = Flask(__name__)

my_models = MyModels()


@app.route("/")
def hello_world():
    return render_template("take_image.html")


@app.route("/predict", methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']

    if file.filename == '':
        return 'No selected file'

    file_stream = io.BytesIO()
    file.save(file_stream)
    file_stream.seek(0)

    selected_model = request.form['model']
    if selected_model == 'mobilenet':
        prediction = my_models.mobilenet_model(file_stream)
    elif selected_model == 'inception':
        prediction = my_models.inception_model(file_stream)
    elif selected_model == 'densenet':
        prediction = my_models.densenet_model(file_stream)
    else:
        prediction = 'Invalid model selection'

    return f"<h2>{selected_model.capitalize()} Prediction: {prediction}</h2>"


# @app.route("/densenet", methods=['POST'])
# def densenet():
#     if 'image' not in request.files:
#         return 'No file part'

#     file = request.files['image']

#     if file.filename == '':
#         return 'No selected file'

#     file_stream = io.BytesIO()
#     file.save(file_stream)
#     file_stream.seek(0)
#     return f"<h2>Densenet: {my_models.densenet_model(file_stream)}</h2>"


# @app.route("/inception", methods=['POST'])
# def inception():
#     if 'image' not in request.files:
#         return 'No file part'

#     file = request.files['image']

#     if file.filename == '':
#         return 'No selected file'

#     file_stream = io.BytesIO()
#     file.save(file_stream)
#     file_stream.seek(0)
#     return f"<h2>Inception: {my_models.inception_model(file_stream)}</h2>"


# @app.route("/mobilenet", methods=['POST'])
# def mobilenet():
#     if 'image' not in request.files:
#         return 'No file part'

#     file = request.files['image']

#     if file.filename == '':
#         return 'No selected file'

#     file_stream = io.BytesIO()
#     file.save(file_stream)
#     file_stream.seek(0)
#     return f"<h2>MobileNet: {my_models.mobilenet_model(file_stream)}</h2>"


app.run(debug=True)
