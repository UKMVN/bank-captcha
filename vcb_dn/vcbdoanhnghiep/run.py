from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from flask import jsonify

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_json
import base64
import time

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

img_width = 145
img_height = 50
max_length = 6
characters_mbbank = ['2', '3', '4', '5', '6', '7', '8', '9', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z']
# Mapping characters to integers
char_to_num = layers.StringLookup(
    vocabulary=list(characters_mbbank), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def LoadModel(file):
    # JSON format
    xfile = os.path.splitext(file)
    if (xfile[1] == ".json"):
        with open(file, "r") as json:
            json_model = json.read()
        model = keras.models.model_from_json(json_model)
        model.load_weights(xfile[0] + ".wgt")
    # ONNX format
    elif (xfile[1] == ".onnx"):
        raise Exception("LoadModel; ONNX format not supported yet")
        model = None
    # TF/Keras format
    else:
        model = keras.models.load_model(file, custom_objects={'leaky_relu': tf.nn.leaky_relu})
    return model

model_mbbank = LoadModel("vcb.json")

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text
  
def encode_base64x(base64):
    img = tf.io.decode_base64(base64)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    return {"image": img}

@app.route("/api/captcha/vcbdigibiz", methods=["POST"])
@cross_origin(origin='*')
def vcb():
    content = request.json
    start_time = time.time()
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"))["image"]
    listImage = np.array([image_encode])
    preds = model_mbbank.predict(listImage)
    pred_texts = decode_batch_predictions(preds)
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status = "success",captcha = captcha)

    return response


# Chạy server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8265')  # -> chú ý port, không để bị trùng với port chạy cái khác