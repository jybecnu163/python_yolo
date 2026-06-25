import io
import requests
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
model = YOLO("yolo11n.pt")


def predict_image(img):
    results = model(img)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls)
            class_name = model.names[cls_id]
            bbox = box.xyxy.cpu().numpy().flatten().tolist()
            detections.append({
                'class': cls_id,
                'class_name': class_name,
                'confidence': float(box.conf),
                'bbox': [float(x) for x in bbox]
            })
    return detections


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>YOLO 目标检测测试</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 20px auto; padding: 0 20px; }
        .section { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 8px; }
        .section h2 { margin-top: 0; }
        .upload-area, .url-area { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
        .upload-area input[type="file"] { flex: 1; }
        .url-area input[type="text"] { flex: 3; min-width: 250px; padding: 8px; }
        button { padding: 8px 20px; background: #007bff; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
        button:disabled { opacity: 0.6; }
        .status { margin-left: 15px; font-weight: bold; }
        .preview { position: relative; display: inline-block; max-width: 100%; margin-top: 15px; }
        #canvas { max-width: 100%; border: 1px solid #ddd; }
        #result-list { margin-top: 20px; }
        .det-item { background: #f5f5f5; padding: 8px 12px; margin: 4px 0; border-radius: 4px; }
        .det-item .cls { font-weight: bold; color: #007bff; }
        .det-item .conf { color: #28a745; }
        .error { color: red; }
        .tabs { display: flex; gap: 10px; margin-bottom: 15px; }
        .tabs button { background: #eee; color: #333; border: 1px solid #ccc; }
        .tabs button.active { background: #007bff; color: #fff; border-color: #007bff; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <h1>🧠 YOLO 目标检测测试</h1>

    <div class="tabs">
        <button id="tabFile" class="active" onclick="switchTab('file')">📁 上传图片</button>
        <button id="tabUrl" onclick="switchTab('url')">🌐 图片URL</button>
    </div>

    <div id="tabFileContent" class="tab-content active">
        <div class="section">
            <h2>从本地上传</h2>
            <div class="upload-area">
                <input type="file" id="imageInput" accept="image/*">
                <button id="detectFileBtn">检测</button>
                <span id="statusFile" class="status"></span>
            </div>
        </div>
    </div>

    <div id="tabUrlContent" class="tab-content">
        <div class="section">
            <h2>从网络地址</h2>
            <div class="url-area">
                <input type="text" id="imageUrlInput" placeholder="请输入图片URL，例如 https://example.com/image.jpg">
                <button id="detectUrlBtn">检测</button>
                <span id="statusUrl" class="status"></span>
            </div>
        </div>
    </div>

    <div class="preview">
        <canvas id="canvas"></canvas>
    </div>
    <div id="result-list"></div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const resultList = document.getElementById('result-list');
        let currentImage = null;

        function switchTab(tab) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tabs button').forEach(el => el.classList.remove('active'));
            if (tab === 'file') {
                document.getElementById('tabFileContent').classList.add('active');
                document.getElementById('tabFile').classList.add('active');
            } else {
                document.getElementById('tabUrlContent').classList.add('active');
                document.getElementById('tabUrl').classList.add('active');
            }
            clearResults();
        }

        function clearResults() {
            resultList.innerHTML = '';
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            currentImage = null;
        }

        function drawImageOnCanvas(img) {
            const maxWidth = 800;
            let w = img.width, h = img.height;
            if (w > maxWidth) {
                h = h * (maxWidth / w);
                w = maxWidth;
            }
            canvas.width = w;
            canvas.height = h;
            ctx.drawImage(img, 0, 0, w, h);
            currentImage = img;
        }

        function drawDetections(detections) {
            if (!currentImage) return;
            drawImageOnCanvas(currentImage);

            const w = canvas.width, h = canvas.height;
            const colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080', '#FFC0CB', '#00FFFF', '#FFD700'];

            detections.forEach((det, idx) => {
                let bbox = det.bbox;
                if (!Array.isArray(bbox)) return;
                if (bbox.length === 1 && Array.isArray(bbox[0])) bbox = bbox[0];
                if (bbox.length < 4) return;
                const [x1, y1, x2, y2] = bbox.map(v => parseFloat(v));
                if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) return;

                const scaleX = canvas.width / currentImage.width;
                const scaleY = canvas.height / currentImage.height;
                const left = x1 * scaleX, top = y1 * scaleY;
                const right = x2 * scaleX, bottom = y2 * scaleY;

                const color = colors[idx % colors.length];
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(left, top, right - left, bottom - top);

                const label = `${det.class_name} ${(det.confidence * 100).toFixed(1)}%`;
                ctx.fillStyle = color;
                ctx.font = '16px Arial';
                const textWidth = ctx.measureText(label).width;
                ctx.fillRect(left, top - 22, textWidth + 8, 22);
                ctx.fillStyle = '#fff';
                ctx.fillText(label, left + 4, top - 4);
            });
        }

        function renderResultList(detections) {
            let html = '<h3>检测结果</h3>';
            if (detections.length === 0) {
                html += '<p>未检测到目标</p>';
            } else {
                detections.forEach((d, i) => {
                    let bbox = d.bbox;
                    if (!Array.isArray(bbox)) bbox = [];
                    if (bbox.length === 1 && Array.isArray(bbox[0])) bbox = bbox[0];
                    const bboxStr = bbox.map(v => {
                        const num = parseFloat(v);
                        return isNaN(num) ? '0' : num.toFixed(0);
                    }).join(', ');

                    html += `<div class="det-item">
                        <span class="cls">${d.class_name}</span>
                        置信度 <span class="conf">${(d.confidence * 100).toFixed(1)}%</span>
                        框: [${bboxStr}]
                    </div>`;
                });
            }
            resultList.innerHTML = html;
        }

        // ---------- 加载网络图片到画布 ----------
        function loadImageFromUrl(url) {
            return new Promise((resolve, reject) => {
                const img = new Image();
                // 设置 crossOrigin 以避免某些网站的 CORS 问题
                img.crossOrigin = "anonymous";
                img.onload = function() {
                    drawImageOnCanvas(img);
                    resolve(img);
                };
                img.onerror = function() {
                    reject(new Error('图片加载失败，请检查URL是否有效'));
                };
                img.src = url;
            });
        }

        // ---------- 文件上传检测 ----------
        document.getElementById('detectFileBtn').addEventListener('click', function() {
            const file = document.getElementById('imageInput').files[0];
            if (!file) {
                alert('请先选择一张图片');
                return;
            }
            const reader = new FileReader();
            reader.onload = function(ev) {
                const img = new Image();
                img.onload = function() {
                    drawImageOnCanvas(img);
                    resultList.innerHTML = '';
                };
                img.src = ev.target.result;
            };
            reader.readAsDataURL(file);

            const formData = new FormData();
            formData.append('image', file);
            const statusEl = document.getElementById('statusFile');
            statusEl.textContent = '检测中...';
            statusEl.style.color = '#333';

            fetch('/detect', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error || '检测失败'); });
                }
                return response.json();
            })
            .then(data => {
                statusEl.textContent = `✅ 检测完成，共 ${data.detections.length} 个目标`;
                statusEl.style.color = '#28a745';
                renderResultList(data.detections);
                drawDetections(data.detections);
            })
            .catch(err => {
                statusEl.textContent = '❌ ' + err.message;
                statusEl.style.color = 'red';
                resultList.innerHTML = `<div class="error">错误：${err.message}</div>`;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            });
        });

        // ---------- URL 检测（修复版） ----------
        document.getElementById('detectUrlBtn').addEventListener('click', function() {
            const url = document.getElementById('imageUrlInput').value.trim();
            if (!url) {
                alert('请输入图片URL');
                return;
            }

            const statusEl = document.getElementById('statusUrl');
            const btn = this;
            btn.disabled = true;
            statusEl.textContent = '加载图片中...';
            statusEl.style.color = '#333';

            // 1. 先加载图片并显示到画布
            loadImageFromUrl(url)
                .then(img => {
                    // 图片已显示，currentImage 已设置
                    statusEl.textContent = '图片加载成功，正在检测...';
                    // 2. 发送检测请求
                    return fetch('/detect_url', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image_url: url })
                    });
                })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(err => { throw new Error(err.error || '检测失败'); });
                    }
                    return response.json();
                })
                .then(data => {
                    statusEl.textContent = `✅ 检测完成，共 ${data.detections.length} 个目标`;
                    statusEl.style.color = '#28a745';
                    renderResultList(data.detections);
                    drawDetections(data.detections);
                })
                .catch(err => {
                    statusEl.textContent = '❌ ' + err.message;
                    statusEl.style.color = 'red';
                    resultList.innerHTML = `<div class="error">错误：${err.message}</div>`;
                    // 如果图片加载失败，画布保持清空
                })
                .finally(() => {
                    btn.disabled = false;
                });
        });

        // 回车触发检测
        document.getElementById('imageUrlInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                document.getElementById('detectUrlBtn').click();
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/detect', methods=['POST'])
def detect_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({'error': 'Invalid image file'}), 400
    detections = predict_image(img)
    return jsonify({'detections': detections})


@app.route('/detect_url', methods=['POST'])
def detect_url():
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({'error': 'Missing image_url'}), 400
    url = data['image_url']
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return jsonify({'error': 'URL does not point to an image'}), 400
        img = Image.open(io.BytesIO(response.content))
    except Exception as e:
        return jsonify({'error': f'Failed to load image: {str(e)}'}), 400
    detections = predict_image(img)
    return jsonify({'detections': detections})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
