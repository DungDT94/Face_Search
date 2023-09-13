from flask import Flask
from flask import request
from PIL import Image
import requests
from app import *
import cv2
from config import feature, name

app = Flask(__name__)
model_face = FaceSearch(feature, name, 1.1)


def create_respone(result_json, errorCode, errorMessage):
    result = {
        'data': result_json,
        'errorCode': errorCode,
        'errorMessage': errorMessage
    }
    return result


@app.route("/hello")
def hello():
    return "Hello, Welcome to Dung Dinh"


@app.route("/search", methods=['POST', 'GET'])
def predict():
    format_type = request.args.get('format_type', default='file')
    ret = {}
    if format_type in ['file', 'url']:
        if format_type == 'file':
            try:
                # image = Image.open(io.BytesIO(request.files["img1"].read()))
                uploaded_file = request.files["img1"]
                image_data = uploaded_file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except:
                ret = create_respone({}, '3', 'Incorrect image format')
                return ret
        else:
            try:
                img1 = request.args.get('url')
                image = Image.open(requests.get(img1, stream=True).raw)
            except:
                ret = create_respone("", '2', 'Url is unavailable')
                return ret

        ret['data'] = model_face.process(image)
        if ret['data'] is not None:
            ret['errorCode'] = '0'
            ret['errorMessage'] = 'Success'
        else:
            ret = create_respone("", '1', 'The photo does not contain any person')
    else:
        ret = create_respone("", '6', 'Incorrect format type')
    return ret


@app.route("/add", methods=['POST', 'GET'])
def add():
    name = request.args.get('name')
    format_type = request.args.get('format_type', default='file')
    ret = {}
    if format_type in ['file', 'url']:
        if format_type == 'file':
            try:
                # image = Image.open(io.BytesIO(request.files["img1"].read()))
                uploaded_file = request.files["img1"]
                image_data = uploaded_file.read()
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except:
                ret = create_respone({}, '3', 'Incorrect image format')
                return ret
        else:
            try:
                url = request.args.get('url')
                response = requests.get(url, stream=True)
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            except:
                ret = create_respone("", '2', 'Url is unavailable')
                return ret
        try:
            model_face.add_1user(image, name)
            ret = create_respone("", '0', 'add successfully')
        except:
            ret = create_respone("", '1', 'The photo does not contain any person')
    else:
        ret = create_respone("", '6', 'Incorrect format type')
    return ret


@app.route("/show", methods=['GET'])
def show():
    ret = model_face.show()
    return ret


@app.route("/delete", methods=['POST', 'GET'])
def delete():
    name = request.args.get('name')
    try:
        model_face.delete(name)
        ret = create_respone("", '0', 'delete successfully')
    except:
        ret = create_respone("", '1', 'delete failed')
    return ret


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)
