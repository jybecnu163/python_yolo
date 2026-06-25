conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/

conda config --show channels
conda create -n yolo_tf python=3.10
conda activate yolo_tf
pip install ultralytics tensorflow

pip install --upgrade ultralytics
pip install tf_keras<=2.19.0 sng4onnx>=1.0.1 onnx_graphsurgeon>=0.3.26 ai-edge-litert>=1.2.0

pip install protobuf==7.35.1
yolo export model=yolo11n.pt format=tflite int8 data=coco128.yaml
