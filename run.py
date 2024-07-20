import io

import cv2
from PIL import Image
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from flask import jsonify

import os

import pytesseract

# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

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

characters_bidv = ['6', 'y', 'V', 'm', 'r', '8', 'u', 'x', '5', 'v', 'X', '3', 'h', 'f', '-', '9', 'q', 's', 'k', 'a',
                   'j', '7', 't', 'g', '4', 'b', 'z', 'p', 'n', 'c', '2', 'd', 'e']
characters_vcb = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
characters_garena = ['l', '2', '5', 'v', 'e', 'f', '0', 'w', '4', '8', '-', 'u', 'd', '3', 'p', 'x', 'a', 'z', '6', 't',
                     'm', 'b', 'k', 's', 'n', '9', 'r', 'h']
characters_viettel = ['8', 'a', '6', '9', 'g', 'x', 'j', '5', '7', 'f', '3', '2', 'e', 'd', '4', 'u', 'p', 'z', 't',
                      'l', 'y', 'k', 'c', 'b', 'n', 's', 'r', 'h', 'm']
characters_mb = ['K', 'M', 'C', 'e', 'g', 'k', 'u', 'z', 't', '3', 'U', 'a', '5', 'A', 'y', 'H', 'q', 'Z', 'V', '7',
                 'Q', '2', '4', 'Y', '-', 'h', '8', 'v', '6', 'd', 'b', 'n', 'p', 'P', 'E', 'c', 'm', 'D', 'B', '9',
                 'N', 'G']
characters_bangiftcode = ['b', '8', '0', '4', '-', 'e', 'c', '6', '5', 'f', '7', '2', '9', 'd', '1', 'a', '3']
characters_bidv_digibiz = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
characters_vcb_digibiz = ['2', '3', '4', '5', '6', '7', '8', '9', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z']

img_width = 320
img_height = 80

# Số lượng tối đa trong captcha ( dài nhất là 6)
max_length = 15

char_to_num_vcb = layers.StringLookup(vocabulary=list(characters_vcb), mask_token=None)
char_to_num_bidv = layers.StringLookup(vocabulary=list(characters_bidv), mask_token=None)
char_to_num_viettel = layers.StringLookup(vocabulary=list(characters_viettel), mask_token=None)
char_to_num_garena = layers.StringLookup(vocabulary=list(characters_garena), mask_token=None)
char_to_num_mb = layers.StringLookup(vocabulary=list(characters_mb), mask_token=None)
char_to_num_bangiftcode = layers.StringLookup(vocabulary=list(characters_bangiftcode), mask_token=None)

num_to_char_vcb = layers.StringLookup(vocabulary=char_to_num_vcb.get_vocabulary(), mask_token=None, invert=True)
num_to_char_bidv = layers.StringLookup(vocabulary=char_to_num_bidv.get_vocabulary(), mask_token=None, invert=True)
num_to_char_viettel = layers.StringLookup(vocabulary=char_to_num_viettel.get_vocabulary(), mask_token=None, invert=True)
num_to_char_garena = layers.StringLookup(vocabulary=char_to_num_garena.get_vocabulary(), mask_token=None, invert=True)
num_to_char_mb = layers.StringLookup(vocabulary=char_to_num_mb.get_vocabulary(), mask_token=None, invert=True)
num_to_char_bangiftcode = layers.StringLookup(vocabulary=char_to_num_bangiftcode.get_vocabulary(), mask_token=None,
                                              invert=True)

char_to_num_bidv_digibiz = layers.StringLookup(vocabulary=list(characters_bidv_digibiz), mask_token=None)
num_to_char_bidv_digibiz = layers.StringLookup(vocabulary=char_to_num_bidv_digibiz.get_vocabulary(), mask_token=None, invert=True)

char_to_num_vcb_digibiz = layers.StringLookup(vocabulary=list(characters_vcb_digibiz), mask_token=None)
num_to_char_vcb_digibiz = layers.StringLookup(vocabulary=char_to_num_vcb_digibiz.get_vocabulary(), mask_token=None, invert=True)


# Đọc ảnh base64 và mã hóa
def encode_base64x(base64):
    img = tf.io.decode_base64(base64)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    return {"image": img}


def encode_base64x_vcb_quannguyen(base64image):
    img = tf.io.decode_base64(base64image)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [50, 155])
    img = tf.transpose(img, perm=[1, 0, 2])
    return {"image": img}
# Dịch từ mã máy thành chữ
def decode_batch_predictions(pred, type):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = []
    for res in results:
        if type in "vcb":
            res = tf.strings.reduce_join(num_to_char_vcb(res)).numpy().decode("utf-8")
        elif type in "bidv":
            res = tf.strings.reduce_join(num_to_char_bidv(res)).numpy().decode("utf-8")
        elif type in "mb":
            res = tf.strings.reduce_join(num_to_char_mb(res)).numpy().decode("utf-8")
        elif type in "garena":
            res = tf.strings.reduce_join(num_to_char_garena(res)).numpy().decode("utf-8")
        elif type in "bangiftcode":
            res = tf.strings.reduce_join(num_to_char_bangiftcode(res)).numpy().decode("utf-8")
        elif type in "vcbdigibiz":
            input_len = np.ones(pred.shape[0]) * pred.shape[1]
            results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
            for res in results:
                res = tf.strings.reduce_join(num_to_char_vcb_digibiz(res)).numpy().decode("utf-8")
                output_text.append(res)
            return output_text
        elif type in "bidvdigibiz":
            input_len = np.ones(pred.shape[0]) * pred.shape[1]
            results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
            for res in results:
                res = tf.strings.reduce_join(num_to_char_bidv_digibiz(res)).numpy().decode("utf-8")
                output_text.append(res)
            return output_text
        else:
            res = tf.strings.reduce_join(num_to_char_viettel(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


# Tạo class CTCLayer | DichVuDark.Vn
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost


# load model vcb

# json_file_vcb = open('model_vcb.json', 'r')
# loaded_model_json = json_file_vcb.read()
# json_file_vcb.close()
# loaded_model_vcb = model_from_json(loaded_model_json)
# loaded_model_vcb.load_weights("model_vcb.h5")
# loaded_model_vcb.load_weights("vcb_model_quannguyen.h5")
loaded_model_vcb = keras.models.load_model("vcb_model_quannguyen.h5", custom_objects={"CTCLayer": CTCLayer})
prediction_model = keras.models.Model(
    loaded_model_vcb.get_layer(name="image").input, loaded_model_vcb.get_layer(name="dense2").output
)

# load model bidv
json_file_bidv = open('model_bidv.json', 'r')
loaded_model_json = json_file_bidv.read()
json_file_bidv.close()
loaded_model_bidv = model_from_json(loaded_model_json)
loaded_model_bidv.load_weights("model_bidv.h5")

# load model viettel
json_file_viettel = open('model_viettel.json', 'r')
loaded_model_json = json_file_viettel.read()
json_file_viettel.close()
loaded_model_viettel = model_from_json(loaded_model_json)
loaded_model_viettel.load_weights("model_viettel.h5")

# load model garena
json_file_garena = open('model_garena.json', 'r')
loaded_model_json = json_file_garena.read()
json_file_garena.close()
loaded_model_garena = model_from_json(loaded_model_json)
loaded_model_garena.load_weights("model_garena.h5")

# load model mb
json_file_mb = open('model_mb.json', 'r')
loaded_model_json = json_file_mb.read()
json_file_mb.close()
loaded_model_mb = model_from_json(loaded_model_json)
loaded_model_mb.load_weights("model_mb.h5")

# load model mb
json_file_bangiftcode = open('model_bangiftcode.json', 'r')
loaded_model_json = json_file_bangiftcode.read()
json_file_bangiftcode.close()
loaded_model_bangiftcode = model_from_json(loaded_model_json)
loaded_model_bangiftcode.load_weights("model_bangiftcode.h5")


def load_model_sieuthicode(file):
    # JSON format
    xfile = os.path.splitext(file)
    if xfile[1] == ".json":
        with open(file, "r") as json:
            json_model = json.read()
        model = keras.models.model_from_json(json_model)
        model.load_weights(xfile[0] + ".wgt")
    # ONNX format
    elif xfile[1] == ".onnx":
        raise Exception("LoadModel; ONNX format not supported yet")
        model = None
    # TF/Keras format
    else:
        model = keras.models.load_model(file, custom_objects={'leaky_relu': tf.nn.leaky_relu})
    return model

# hàm để truy cập: 127.0.0.1/run -> 127.0.0.1 là ip server
@app.route("/api/captcha/vietcombank", methods=["POST"])
@cross_origin(origin='*')
def vcb():
    content = request.json
    imgstring = content['base64']
    image_encode = encode_base64x_vcb_quannguyen(imgstring.replace("+", "-").replace("/", "_"))["image"]
    listImage = np.array([image_encode])
    preds = prediction_model.predict(listImage)
    pred_texts = decode_batch_predictions(preds, 'vcb')
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status="success", captcha=captcha)
    return response


# hàm để truy cập: 127.0.0.1/run -> 127.0.0.1 là ip server
@app.route("/api/captcha/bidv", methods=["POST"])
@cross_origin(origin='*')
def bidv():
    content = request.json
    start_time = time.time()
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"))["image"]
    listImage = np.array([image_encode])
    preds = loaded_model_bidv.predict(listImage)
    pred_texts = decode_batch_predictions(preds, "bidv")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status=True, captcha=captcha)

    return response


# hàm để truy cập: 127.0.0.1/run -> 127.0.0.1 là ip server
@app.route("/api/captcha/viettel", methods=["POST"])
@cross_origin(origin='*')
def viettel():
    content = request.json
    start_time = time.time()
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"))["image"]
    listImage = np.array([image_encode])
    preds = loaded_model_viettel.predict(listImage)
    pred_texts = decode_batch_predictions(preds, "viettel")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status="success", captcha=captcha)

    return response


# hàm để truy cập: 127.0.0.1/run -> 127.0.0.1 là ip server
@app.route("/api/captcha/garena", methods=["POST"])
@cross_origin(origin='*')
def garena():
    content = request.json
    start_time = time.time()
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"))["image"]
    listImage = np.array([image_encode])
    preds = loaded_model_garena.predict(listImage)
    pred_texts = decode_batch_predictions(preds, "garena")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status="success", captcha=captcha)

    return response


# hàm để truy cập: 127.0.0.1/run -> 127.0.0.1 là ip server
@app.route("/api/captcha/mb", methods=["POST"])
@cross_origin(origin='*')
def mb():
    content = request.json
    start_time = time.time()
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"))["image"]
    listImage = np.array([image_encode])
    preds = loaded_model_mb.predict(listImage)
    pred_texts = decode_batch_predictions(preds, "mb")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status="success", captcha=captcha)

    return response


# hàm để truy cập: 127.0.0.1/run -> 127.0.0.1 là ip server
@app.route("/api/captcha/bangiftcode", methods=["POST"])
@cross_origin(origin='*')
def bangiftcode():
    content = request.json
    start_time = time.time()
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"))["image"]
    listImage = np.array([image_encode])
    preds = loaded_model_bangiftcode.predict(listImage)
    pred_texts = decode_batch_predictions(preds, "bangiftcode")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status="success", captcha=captcha)

    return response


@app.route("/api/digital_otp", methods=["POST"])
@cross_origin(origin='*')
def digital_otp_mbbank():
    content = request.json
    imgstring = content['base64']
    image_decode = base64.b64decode(imgstring)
    image = Image.open(io.BytesIO(image_decode))
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    otp = pytesseract.image_to_string(image_gray, lang='eng',
                                      config='-c tessedit_char_whitelist=0123456789')
    return jsonify(status="success", otp=otp.strip())


@app.route("/api/captcha/msb", methods=["POST"])
@cross_origin()
def digital_otp_msb():
    content = request.json
    image_data = base64.b64decode(content['base64'])
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((515, 180))
    image = image.convert('L')
    otp = pytesseract.image_to_string(image, lang='eng',
                                      config='-c tessedit_char_whitelist=0123456789')
    return jsonify(status=True, captcha=otp.strip())


model_vcbdigibiz = load_model_sieuthicode("./vcb_dn/vcb.json")
model_bidvdigibiz = load_model_sieuthicode("./dn_bidv/bidv.json")


@app.route("/api/captcha/vcbdigibiz", methods=["POST"])
@cross_origin()
def vcb_vcbdigibiz():
    content = request.json
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"))["image"]
    listImage = np.array([image_encode])
    preds = model_vcbdigibiz.predict(listImage)
    pred_texts = decode_batch_predictions(preds, "vcbdigibiz")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status=True, captcha=captcha)
    return response


@app.route("/api/captcha/bidvdigibiz", methods=["POST"])
@cross_origin()
def bidv_bidvdigibiz():
    content = request.json
    imgstring = content['base64']
    image_encode = encode_base64x(imgstring.replace("+", "-").replace("/", "_"))["image"]
    listImage = np.array([image_encode])
    preds = model_bidvdigibiz.predict(listImage)
    pred_texts = decode_batch_predictions(preds, "bidvdigibiz")
    captcha = pred_texts[0].replace('[UNK]', '').replace('-', '')
    response = jsonify(status=True, captcha=captcha)
    return response


# Chạy server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8122')  # -> chú ý port, không để bị trùng với port chạy cái khác
