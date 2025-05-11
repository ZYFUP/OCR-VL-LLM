from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from dgocr.dgocr import DGOCR
import base64
import requests
from typing import Dict, Any
from dataclasses import dataclass, field
from copy import deepcopy
import paddle
app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.DEBUG)

# 配置上传和结果文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# OCR模型初始化
rec_path = os.path.abspath("./models/base_seglink++/recognition_model_general")
det_path = os.path.abspath("./models/base_seglink++/detection_model_general/model_1024x1024.onnx")
app.logger.info(f"Using recognition model at: {rec_path}")
app.logger.info(f"Using detection model at: {det_path}")

try:
    ocr = DGOCR(rec_path, det_path, img_size=1024, model_type="seglink", cpu_thread_num=12)
except Exception as e:
    app.logger.error(f"OCR初始化失败: {str(e)}")
    raise


# API配置基础类
@dataclass
class ApiConfig:
    host: str = "10.110.237.138"
    port: str = "3000"
    model: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    headers: Dict[str, str] = field(default_factory=lambda: {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImQ5MWQ3YjRiLTk3OWQtNGU3Yi1hN2MyLWNjZmY5ODRjNTIzNyJ9.8VtmHdivvEqNEqIH7iswtnfy81yHtIzBM-p2dqQBt5k",
        "Content-Type": "application/json"
    })
    temperature: float = 0.0
    stream: bool = False
    api_endpoint: str = "/api/chat/completions"
    request_timeout: int = 60
    image_content_type: str = "image/jpeg"

# 用于文本校验的第三模型配置
@dataclass
class ValidationModelConfig(ApiConfig):
    model: str = "deepseek-ai/DeepSeek-V3"


class ImageProcessor:
    @staticmethod
    def encode_to_base64(image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            app.logger.error("找不到文件")
            raise


class RemoteModelClient:
    def __init__(self, config: ApiConfig):
        self.config = config
        self.base_url = f"http://{config.host}:{config.port}{config.api_endpoint}"

    def build_request_body(self, base64_image: str) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "options": {"temperature": self.config.temperature},
            "stream": self.config.stream,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{self.config.image_content_type};base64,{base64_image}"}
                     }]
            }]
        }

    def send_request(self, image_path: str) -> Dict[str, Any]:
        try:
            base64_image = ImageProcessor.encode_to_base64(image_path)
            request_body = self.build_request_body(base64_image)
            response = requests.post(
                self.base_url,
                json=request_body,
                headers=self.config.headers,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            app.logger.error(f"远程API请求失败: {str(e)}")
            return {"error": str(e)}


@app.route('/ocr', methods=['POST'])
@app.route('/results')
def process_ocr():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 保存上传的文件
        app.logger.debug(f"文件保存中...")
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        app.logger.debug(f"文件保存成功：{upload_path}")
        app.logger.debug(f"图片上传OCR中...")
        app.logger.debug(f"图片上传视觉模型中...")
        # 同时处理OCR和远程API
        ocr_result = ocr.run(upload_path)
        app.logger.debug(f"OCR接收成功！正在处理图片")
        remote_client = RemoteModelClient(ApiConfig())
        app.logger.debug(f"视觉模型接收成功！正在处理图片")
        remote_result = remote_client.send_request(upload_path)

        app.logger.debug(f"OCR识别结果：{ocr_result}")
        app.logger.debug(f"视觉模型识别结果：{remote_result}")

        # 生成结果图片
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        ocr.draw(upload_path, ocr_result, result_path)


        # 格式化OCR结果
        formatted_ocr_result = [
            {
                "id": i,
                "text": item[1][0],
                "confidence": float(item[1][1]),
                "position": item[0]
            } for i, item in enumerate(ocr_result)
        ]

        # 格式化远程模型结果
        formatted_remote_result = ""
        if isinstance(remote_result, dict) and not remote_result.get("error"):
            formatted_remote_result = remote_result.get('choices', [{}])[0].get('message', {}).get('content', "")

        # 拼文本，传校验模型
        app.logger.debug(f"正在生成校验文本...")
        combined_text = f"{' '.join([item['text'] for item in formatted_ocr_result])} ******** {formatted_remote_result}"

        validation_client = RemoteModelClient(ValidationModelConfig())
        validation_payload = {
            "model": validation_client.config.model,
            "options": {"temperature": validation_client.config.temperature},
            "stream": False,
            "messages": [{
                "role": "user",
                "content": combined_text
            }]
        }
        app.logger.debug(f"正在上传校验文本：{combined_text}")
        app.logger.debug(f"上传成功！")
        app.logger.debug(f"正在等待校验模型回应...")
        try:
            validation_response = requests.post(
                validation_client.base_url,
                json=validation_payload,
                headers=validation_client.config.headers,
                timeout=validation_client.config.request_timeout
            )
            validation_response.raise_for_status()
            validation_result = validation_response.json().get("choices", [{}])[0].get("message", {}).get("content", "")

        except Exception as e:
            app.logger.error(f"校验模型请求失败: {str(e)}")
            validation_result = f"校验模型校验失败: {str(e)}"
        app.logger.debug(f"校验结果：{validation_result}")
        app.logger.debug(f"标注图片已保存至：{result_path}")
        app.logger.debug(f"当你看到这句话时，程序已经圆满完成所有既定任务")
        return jsonify({
            'ocr_result': formatted_ocr_result,
            'remote_model_result': formatted_remote_result,
            'validation_result': validation_result,
            'result_image': f"http://{request.host}/results/{result_filename}"
        }), 200

    except Exception as e:
        app.logger.error(f"处理异常：{str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'ocr_result': None,
            'remote_model_result': None,
            'validation_result': None
        }), 500


@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='10.110.237.100', port=5000, debug=True)
