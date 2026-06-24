Flask YOLO Object Detection API
一个基于 Flask 和 Ultralytics YOLO 的 RESTful 目标检测服务，提供简洁的 Web 测试界面，支持本地图片上传和网络图片 URL 两种检测方式。检测结果以 JSON 格式返回，同时前端页面会实时在图片上绘制边界框并显示类别与置信度。

✨ 特性
🚀 双模式检测：支持上传本地图片文件（multipart/form-data）和通过 URL 加载网络图片（application/json）。

🎨 可视化测试页面：开箱即用的 Web UI，可切换“上传图片”和“图片 URL”两种输入方式，检测后自动在画布上绘制边界框并列出结果。

📦 轻量级依赖：基于 Flask 和 Ultralytics，快速部署。

🔌 可扩展：易于替换 YOLO 模型（如 yolo11n.pt、yolov8s.pt 等），或集成到更大的应用中。

🛡️ 错误处理：完善的异常捕获和友好错误提示。

🛠 技术栈
Python 3.10+（推荐 3.10，最高支持 3.13）

Flask – Web 框架

Ultralytics YOLO – 目标检测模型

Pillow – 图像处理

Requests – 下载网络图片

PyTorch – 深度学习框架（Ultralytics 依赖）

📦 安装
1. 克隆或下载项目
bash
git clone <your-repo-url>
cd flask-yolo-api
2. 创建虚拟环境（推荐）
bash
# 使用 conda
conda create -n yolo python=3.10
conda activate yolo

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
3. 安装 PyTorch（根据硬件选择）
访问 PyTorch 官网 获取适合你系统的安装命令。例如 CUDA 11.8：

bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
如果仅使用 CPU：

bash
pip install torch torchvision
4. 安装其他依赖
bash
pip install -r requirements.txt
requirements.txt 内容：

txt
ultralytics>=8.0.0
Flask>=2.0.0
Pillow>=9.0.0
requests>=2.25.0
numpy>=1.21.0
opencv-python>=4.5.0
🚀 快速启动
bash
python app.py
服务默认运行在 http://127.0.0.1:5000。打开浏览器访问该地址，即可看到测试页面。

📡 API 接口
1. 文件上传检测
端点：POST /detect

Content-Type：multipart/form-data

参数：

字段	类型	描述
image	文件（图片）	支持的格式：JPG, PNG, JPEG 等
响应示例：

json
{
  "detections": [
    {
      "class": 0,
      "class_name": "person",
      "confidence": 0.92,
      "bbox": [120.5, 45.3, 310.8, 480.2]
    },
    {
      "class": 56,
      "class_name": "chair",
      "confidence": 0.78,
      "bbox": [400.1, 200.0, 550.3, 500.6]
    }
  ]
}
bbox：[x1, y1, x2, y2] 像素坐标（相对于原始图像尺寸）

2. URL 图片检测
端点：POST /detect_url

Content-Type：application/json

请求体：

json
{
  "image_url": "https://example.com/image.jpg"
}
响应：同 /detect

🖥️ 前端测试页面
访问根路径 / 即可使用：

📁 上传图片：点击选择文件，自动预览，点击“检测”得到结果。

🌐 图片URL：输入网络图片地址，点击“检测”或按回车，图片会自动加载并识别。

检测完成后，图片上会绘制彩色边界框，下方列表显示每个目标的类别、置信度和坐标。

📁 项目结构
text
.
├── app.py              # 主程序（包含 Flask 路由、模型加载、前端页面）
├── requirements.txt    # 依赖清单
├── README.md           # 本文档
└── .gitignore          # （可选）忽略文件
⚙️ 配置与优化
更换 YOLO 模型
在 app.py 中修改加载的模型文件：

python
model = YOLO("yolo11n.pt")  # 可替换为 "yolov8s.pt", "yolov8m.pt" 等
首次运行时会自动下载指定的模型权重。

调整检测阈值
可在 predict_image() 中添加置信度过滤，例如：

python
for box in r.boxes:
    if box.conf < 0.5:   # 只保留置信度 ≥0.5 的目标
        continue
    # ...
生产环境部署
建议关闭 debug 模式，并使用 Gunicorn 等 WSGI 服务器：

python
app.run(host='0.0.0.0', port=5000, debug=False)
⚠️ 注意事项
PyTorch 版本：务必根据你的硬件（CPU / CUDA）正确安装，否则可能影响推理速度或报错。

网络图片跨域：前端加载 URL 图片时设置了 crossOrigin = "anonymous"，但部分服务器可能不支持跨域，此时图片可能无法显示（但后端仍能正常检测）。建议使用允许跨域的图片链接。

安全限制：生产环境中，/detect_url 接口应限制允许的域名或添加白名单，防止 SSRF 攻击。

超时设置：网络图片下载默认超时 10 秒，可根据网络情况调整 requests.get(timeout=...)。

📄 许可证
本项目仅供学习和研究使用。YOLO 模型权重遵循其各自的许可证（如 GPL-3.0）。

🙋 反馈与贡献
欢迎提交 Issue 或 Pull Request，让项目变得更好！

Happy detecting! 🎯