import io

from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolo11n.pt")  # 或 yolo26n.pt 等

# ---------- 主页测试页面 ----------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>YOLO 目标检测测试</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 20px auto; padding: 0 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin-bottom: 20px; }
        .preview { position: relative; display: inline-block; max-width: 100%; }
        #canvas { max-width: 100%; border: 1px solid #ddd; }
        #result-list { margin-top: 20px; }
        .det-item { background: #f5f5f5; padding: 8px 12px; margin: 4px 0; border-radius: 4px; }
        .det-item .cls { font-weight: bold; color: #007bff; }
        .det-item .conf { color: #28a745; }
        .error { color: red; }
        button { padding: 10px 24px; background: #007bff; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
        button:disabled { opacity: 0.6; }
        #status { margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🧠 YOLO 目标检测测试</h1>
    <div class="upload-area">
        <input type="file" id="imageInput" accept="image/*">
        <br><br>
        <button id="detectBtn">检测</button>
        <span id="status" style="margin-left: 15px;"></span>
    </div>
    <div class="preview">
        <canvas id="canvas"></canvas>
    </div>
    <div id="result-list"></div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const fileInput = document.getElementById('imageInput');
        const detectBtn = document.getElementById('detectBtn');
        const status = document.getElementById('status');
        const resultList = document.getElementById('result-list');

        let currentImage = null;  // 存储 Image 对象

        // 预览图片并清空旧结果
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(ev) {
                const img = new Image();
                img.onload = function() {
                    currentImage = img;
                    drawImageOnCanvas(img);
                    resultList.innerHTML = ''; // 清空旧结果
                };
                img.src = ev.target.result;
            };
            reader.readAsDataURL(file);
        });

        function drawImageOnCanvas(img) {
            // 缩放图片以适应画布，保持宽高比，最大宽度 800px
            const maxWidth = 800;
            let w = img.width, h = img.height;
            if (w > maxWidth) {
                h = h * (maxWidth / w);
                w = maxWidth;
            }
            canvas.width = w;
            canvas.height = h;
            ctx.drawImage(img, 0, 0, w, h);
        }

        // ---------- 显示检测结果列表 ----------
function renderResultList(detections) {
    let html = '<h3>检测结果</h3>';
    if (detections.length === 0) {
        html += '<p>未检测到目标</p>';
    } else {
        detections.forEach((d, i) => {
            // 安全处理 bbox
            let bbox = d.bbox;
            if (!Array.isArray(bbox)) bbox = [];
            // 如果 bbox 是嵌套数组（例如 [[x1,y1,x2,y2]]），则取第一个
            if (bbox.length === 1 && Array.isArray(bbox[0])) bbox = bbox[0];
            // 确保每个元素是数字
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
    document.getElementById('result-list').innerHTML = html;
}

// ---------- 绘制边界框 ----------
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
        // 转换为数字
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

        // 执行检测
        detectBtn.addEventListener('click', function() {
            const file = fileInput.files[0];
            if (!file) {
                alert('请先选择一张图片');
                return;
            }
            detectBtn.disabled = true;
            status.textContent = '检测中...';

            const formData = new FormData();
            formData.append('image', file);

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
                status.textContent = `✅ 检测完成，共 ${data.detections.length} 个目标`;
                // 显示列表
                let html = '<h3>检测结果</h3>';
                if (data.detections.length === 0) {
                    html += '<p>未检测到目标</p>';
                } else {
                    data.detections.forEach((d, i) => {
                        html += `<div class="det-item">
                            <span class="cls">${d.class_name}</span>
                            置信度 <span class="conf">${(d.confidence * 100).toFixed(1)}%</span>
                            框: [${d.bbox.map(v => v.toFixed(0)).join(', ')}]
                        </div>`;
                    });
                }
              //   resultList.innerHTML = html;
             //    // 绘制边界框
             //    drawDetections(data.detections);
                // 在 fetch 的 then 中
                const detections = data.detections;
                renderResultList(detections);
                drawDetections(detections);
            })
            .catch(err => {
                status.textContent = '❌ ' + err.message;
                resultList.innerHTML = `<div class="error">错误：${err.message}</div>`;
            })
            .finally(() => {
                detectBtn.disabled = false;
            });
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({'error': 'Invalid image file'}), 400

    results = model(img)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls)
            class_name = model.names[cls_id]
            # 确保 bbox 是扁平的数字列表
            bbox = box.xyxy.cpu().numpy().flatten().tolist()
            # 或者使用 bbox = box.xyxy.tolist()[0] 也可，但加一层保险
            detections.append({
                'class': cls_id,
                'class_name': class_name,
                'confidence': float(box.conf),
                'bbox': [float(x) for x in bbox]  # 确保数字类型
            })
    return jsonify({'detections': detections})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
