import base64
import os
from flask import Flask, render_template, request, jsonify
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# Roboflow istemcisini başlatıyoruz
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="UwRaQ8tzsCcNIqZt8P3n"  # Kendi API anahtarını buraya yaz
)

MODEL_ID = "irem-naz/find-charger-wallet-and-others-1-rfdetr-small-t1"

@app.route('/')
def index():
    # index.html dosyasını tarayıcıda açar
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Tarayıcıdan gelen kamera görüntüsünü alır
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'Resim bulunamadı'}), 400
    
    image_data = data['image']
    
    # Base64 formatındaki resmi Roboflow modeline gönderiyoruz
    try:
        result = CLIENT.infer(image_data, model_id=MODEL_ID)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 8080 portunda çalıştırıyoruz
    app.run(host='0.0.0.0', port=8080, debug=True)