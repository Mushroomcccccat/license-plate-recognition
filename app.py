from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
from paddleocr import PaddleOCR, draw_ocr
import os
from PIL import Image
import base64
import io

# paddlepaddle workround，声明环境变量，允许重复的依赖
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

app = Flask(__name__)
cors = CORS(app)
ocr = PaddleOCR()


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/recognize', methods=['POST'])
@cross_origin()
def recognize():
    if 'file' not in request.files:
        return jsonify({
            'error': 'No image file provided'
        }), 400
    file = request.files['file']
    result = ocr.ocr(file.read())
    print(result[0])
    result = result[0]

    # https://pypi.org/project/paddleocr/    
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]    

    # 创建图片对象
    image = Image.open(file)
    
    im_show = draw_ocr(image, boxes, txts, scores)
    im_show = Image.fromarray(im_show)
    
    # 将图片转换成 base64 返回给前端
    buffered = io.BytesIO()
    im_show.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        'boxes': boxes,
        'txts': txts,
        'scores': scores,
        'image_base64': img_str
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
