lunwen-jiance 使用说明

1. 双击启动
- 直接双击：
  start_realtime_detection.bat
- 启动器会自动检测 `.venv` 和系统 `python`，优先选择依赖完整、能够真正运行的环境

2. 浏览器地址
- 默认会自动打开：
  http://127.0.0.1:5000/

3. 当前程序位置
- 主程序：
  lunwen-jiance\app.py
- 页面模板：
  lunwen-jiance\templates\index.html

4. 当前默认使用的权重
- CASME2:
  casme2_fusion_train2_3class copy\fusion_result_asym_se-1\fusion_resnet18_casme2_final_model_for_gui.pth
- DSME:
  DSME_fusion_train copy\fusion_result_dsme_se2\fusion_resnet18_dsme_final_model_for_gui.pth

5. 人脸检测
- 使用 YOLO:
  models\yolo_face\yolov8n-face.pt
- 如果本地没有，程序会尝试自动下载

6. 如果打不开
- 确认已安装 flask / ultralytics / torch / torchvision / opencv-python / pillow / numpy
- 确认摄像头没有被其他软件占用
- 也可以手动运行：
  .venv\Scripts\python.exe lunwen-jiance\app.py
  或：
  python lunwen-jiance\app.py
