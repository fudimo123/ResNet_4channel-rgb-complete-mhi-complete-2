import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from flask import Flask, render_template, Response, jsonify
from PIL import Image
import math
import os
import sys
import urllib.request

from ultralytics import YOLO

# ================= 配置路径 =================
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_IMPL_DIR = os.path.join(REPO_ROOT, "casme2_fusion_train2_3class copy")
YOLO_FACE_MODEL_DIR = os.path.join(REPO_ROOT, "models", "yolo_face")
YOLO_FACE_MODEL_PATH = os.path.join(YOLO_FACE_MODEL_DIR, "yolov8n-face.pt")
YOLO_FACE_MODEL_URL = "https://raw.githubusercontent.com/IshaFaodail/FaceDetection/main/yolov8n-face.pt"
YOLO_FACE_CONF = 0.35
YOLO_FACE_IMGSZ = 640
YOLO_MIN_FACE_SIZE = 100

if MODEL_IMPL_DIR not in sys.path:
    sys.path.insert(0, MODEL_IMPL_DIR)

from fusion_model_se import create_fusion_model_se

app = Flask(__name__)


# ================= 1. 模型结构 (保持不变) =================
class SPPLayer(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4], pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.pool_sizes = pool_sizes
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()
        for i, level in enumerate(self.pool_sizes):
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0] * level - h + 1) / 2),
                       math.floor((kernel_size[1] * level - w + 1) / 2))
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)
            if i == 0:
                x_flatten = tensor
            else:
                x_flatten = torch.cat((x_flatten, tensor), 1)
        return x_flatten


class ResNetBranch(nn.Module):
    def __init__(self, in_channels=3, feature_dim=512):
        super(ResNetBranch, self).__init__()
        base_model = models.resnet18(pretrained=False)
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False), base_model.bn1, base_model.relu,
                                  base_model.maxpool)
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.spp = SPPLayer(pool_sizes=[1, 2, 4])
        self.project = nn.Sequential(nn.Linear(10752, feature_dim), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.spp(x)
        x = self.project(x)
        return x


class FusionResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(FusionResNet, self).__init__()
        self.rgb_branch = ResNetBranch(in_channels=3, feature_dim=512)
        self.dynamic_branch = ResNetBranch(in_channels=3, feature_dim=512)
        self.fusion_fc = nn.Linear(512 * 2, num_classes)

    def forward(self, x_rgb, x_dynamic):
        feat_rgb = self.rgb_branch(x_rgb)
        feat_dyn = self.dynamic_branch(x_dynamic)
        out = self.fusion_fc(torch.cat((feat_rgb, feat_dyn), dim=1))
        return out


# ================= 2. 初始化 =================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
YOLO_DEVICE = 0 if torch.cuda.is_available() else 'cpu'
CLASS_NAMES = ['Positive', 'Negative', 'Surprise']
MODEL_CONFIGS = {
    "casme2": {
        "display_name": "CASME II",
        "weight_path": r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\casme2_fusion_train2_3class copy\fusion_result_asym_se-1\fusion_resnet18_casme2_final_model_for_gui.pth",
    },
    "dsme": {
        "display_name": "DSME",
        "weight_path": r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\DSME_fusion_train copy\fusion_result_dsme_se2\fusion_resnet18_dsme_final_model_for_gui.pth",
    },
}

models = {}
face_detector = None


def _empty_result(label="Detecting...", confidence=0.0):
    return {
        "label": label,
        "confidence": confidence,
    }


current_result = {
    "status": "loading",
    "face_count": 0,
    "detections": [],
    "primary_model": "dsme",
    "models": {
        key: {
            "display_name": config["display_name"],
            **_empty_result(),
        }
        for key, config in MODEL_CONFIGS.items()
    },
}


def ensure_yolo_face_weights():
    os.makedirs(YOLO_FACE_MODEL_DIR, exist_ok=True)
    if os.path.exists(YOLO_FACE_MODEL_PATH):
        return

    print(f"YOLO face weights not found, downloading to: {YOLO_FACE_MODEL_PATH}")
    try:
        urllib.request.urlretrieve(YOLO_FACE_MODEL_URL, YOLO_FACE_MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "Failed to download YOLO face weights automatically. "
            f"Please place the file at: {YOLO_FACE_MODEL_PATH}"
        ) from e
    if not os.path.exists(YOLO_FACE_MODEL_PATH):
        raise RuntimeError(
            "YOLO face weights download did not produce a local file. "
            f"Expected path: {YOLO_FACE_MODEL_PATH}"
        )
    print("YOLO face weights download completed.")


def load_face_detector():
    global face_detector
    ensure_yolo_face_weights()
    print("-" * 50)
    print(f"Loading YOLO face detector from: {YOLO_FACE_MODEL_PATH}")
    face_detector = YOLO(YOLO_FACE_MODEL_PATH)
    print("✅ YOLO face detector loaded successfully!")


def detect_faces_yolo(frame):
    if face_detector is None:
        return []

    try:
        results = face_detector.predict(
            source=frame,
            conf=YOLO_FACE_CONF,
            imgsz=YOLO_FACE_IMGSZ,
            device=YOLO_DEVICE,
            verbose=False,
        )[0]
    except Exception as e:
        print(f"YOLO face detection failed: {e}")
        return []

    detected_faces = []
    frame_h, frame_w = frame.shape[:2]
    if results.boxes is None:
        return detected_faces

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = max(0, min(int(x1), frame_w - 1))
        y1 = max(0, min(int(y1), frame_h - 1))
        x2 = max(0, min(int(x2), frame_w))
        y2 = max(0, min(int(y2), frame_h))
        w = x2 - x1
        h = y2 - y1
        if w < YOLO_MIN_FACE_SIZE or h < YOLO_MIN_FACE_SIZE:
            continue
        detected_faces.append((x1, y1, w, h))

    return sorted(detected_faces, key=lambda f: f[2] * f[3], reverse=True)


def load_models_globally():
    global models
    print("-" * 50)
    print("开始加载模型...")
    loaded_models = {}
    for model_key, config in MODEL_CONFIGS.items():
        print(f"[{config['display_name']}] Loading weights from: {config['weight_path']}")
        temp_model = create_fusion_model_se(
            num_classes=len(CLASS_NAMES),
            dropout_p=0.5,
            pretrained=False,
            transfer_weights_path=None,
        )
        checkpoint = torch.load(config["weight_path"], map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        temp_model.load_state_dict(new_state_dict, strict=True)
        temp_model.to(device)
        temp_model.eval()
        loaded_models[model_key] = temp_model
        print(f"✅ {config['display_name']} 模型加载成功！")
    models = loaded_models
    current_result["status"] = "active"


load_models_globally()
load_face_detector()

# ================= 3. Web 服务逻辑 (UI 修正版) =================

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.face_states = {}
        self.next_face_id = 1
        self.frame_idx = 0

    def __del__(self):
        self.video.release()

    def _create_face_state(self, center):
        face_id = self.next_face_id
        self.next_face_id += 1
        self.face_states[face_id] = {
            "center": center,
            "prev_gray": None,
            "mhi_buffer": None,
            "last_seen": self.frame_idx,
        }
        return face_id

    def _assign_face_ids(self, faces):
        assignments = []
        used_ids = set()
        match_threshold = 140.0

        for face in faces:
            x, y, w, h = face
            center = (x + w / 2.0, y + h / 2.0)
            best_id = None
            best_dist = float("inf")

            for face_id, state in self.face_states.items():
                if face_id in used_ids:
                    continue
                if self.frame_idx - state["last_seen"] > 15:
                    continue

                prev_center = state["center"]
                dist = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
                if dist < best_dist and dist < match_threshold:
                    best_id = face_id
                    best_dist = dist

            if best_id is None:
                best_id = self._create_face_state(center)
            else:
                self.face_states[best_id]["center"] = center
                self.face_states[best_id]["last_seen"] = self.frame_idx

            assignments.append((best_id, face))
            used_ids.add(best_id)

        stale_ids = [
            face_id
            for face_id, state in self.face_states.items()
            if self.frame_idx - state["last_seen"] > 30
        ]
        for face_id in stale_ids:
            self.face_states.pop(face_id, None)

        return assignments

    def get_frame(self):
        self.frame_idx += 1
        success, frame = self.video.read()
        if not success: return None

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 准备可视化的 Dynamic View 图像 (默认全黑)
        vis_dyn_view = np.zeros((100, 100, 3), dtype=np.uint8)

        faces = detect_faces_yolo(frame)

        if len(faces) > 0:
            face_assignments = self._assign_face_ids(faces)
            detections = []

            for face_id, (x, y, w, h) in face_assignments:
                margin = 30
                x_m = max(0, x - margin)
                y_m = max(0, y - margin)
                w_m = min(frame.shape[1] - x_m, w + 2 * margin)
                h_m = min(frame.shape[0] - y_m, h + 2 * margin)

                face_rgb = frame[y_m:y_m + h_m, x_m:x_m + w_m]
                face_gray = gray[y_m:y_m + h_m, x_m:x_m + w_m]

                cv2.rectangle(frame, (x_m, y_m), (x_m + w_m, y_m + h_m), (0, 255, 0), 2)

                if not models:
                    continue

                try:
                    pil_rgb = Image.fromarray(cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB))
                    tensor_rgb = transform_pipeline(pil_rgb).unsqueeze(0).to(device)

                    resized_gray = cv2.resize(face_gray, (224, 224))
                    face_state = self.face_states[face_id]
                    prev_gray = face_state["prev_gray"]
                    if prev_gray is None or prev_gray.shape != resized_gray.shape:
                        diff = np.zeros_like(resized_gray)
                    else:
                        diff = cv2.absdiff(prev_gray, resized_gray)
                    face_state["prev_gray"] = resized_gray

                    if face_state["mhi_buffer"] is None or face_state["mhi_buffer"].shape != diff.shape:
                        face_state["mhi_buffer"] = np.float32(diff)
                    else:
                        cv2.accumulateWeighted(diff, face_state["mhi_buffer"], 0.5)

                    mhi_uint8 = cv2.convertScaleAbs(face_state["mhi_buffer"])
                    mhi_uint8 = cv2.normalize(mhi_uint8, None, 0, 255, cv2.NORM_MINMAX)

                    if not detections:
                        vis_dyn_view = cv2.cvtColor(cv2.resize(mhi_uint8, (100, 100)), cv2.COLOR_GRAY2BGR)

                    mhi_3ch = cv2.cvtColor(mhi_uint8, cv2.COLOR_GRAY2RGB)
                    pil_dyn = Image.fromarray(mhi_3ch)
                    tensor_dyn = transform_pipeline(pil_dyn).unsqueeze(0).to(device)

                    face_results = {}
                    with torch.no_grad():
                        for model_key, loaded_model in models.items():
                            outputs = loaded_model(tensor_rgb, tensor_dyn)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                            conf, predicted = torch.max(probs, 1)
                            label_idx = predicted.item()
                            face_results[model_key] = {
                                "display_name": MODEL_CONFIGS[model_key]["display_name"],
                                "label": CLASS_NAMES[label_idx],
                                "confidence": round(conf.item() * 100, 2),
                            }

                    detections.append(
                        {
                            "face_id": face_id,
                            "bbox": [int(x_m), int(y_m), int(w_m), int(h_m)],
                            "models": face_results,
                        }
                    )
                except Exception as e:
                    print(f"Error on face {face_id}: {e}")

            if detections:
                primary_detection = detections[0]
                current_result["status"] = "active"
                current_result["face_count"] = len(detections)
                current_result["primary_model"] = "dsme"
                current_result["models"] = primary_detection["models"]
                current_result["detections"] = detections
        else:
            current_result["status"] = "no_face"
            current_result["face_count"] = 0
            current_result["primary_model"] = "dsme"
            current_result["models"] = {
                key: {
                    "display_name": config["display_name"],
                    **_empty_result(label="No Face", confidence=0.0),
                }
                for key, config in MODEL_CONFIGS.items()
            }
            current_result["detections"] = []

        # --- UI 绘制修正 ---

        # 1. 左上角只保留 Dynamic View 小图，不再显示模型名/人数/置信度文字
        # 向内缩进，避免被前端视频容器圆角裁掉，视觉上看起来像“缺了一块”
        dyn_margin_x = 24
        dyn_margin_y = 24
        dyn_box_size = 108
        cv2.rectangle(
            frame,
            (dyn_margin_x - 4, dyn_margin_y - 4),
            (dyn_margin_x + dyn_box_size + 4, dyn_margin_y + dyn_box_size + 4),
            (0, 0, 0),
            -1
        )

        # 2. 绘制 Dynamic View 小图到左上角内缩位置
        if vis_dyn_view is not None:
            vis_dyn_view = cv2.resize(vis_dyn_view, (dyn_box_size, dyn_box_size))
            h_v, w_v, _ = vis_dyn_view.shape
            frame[dyn_margin_y:dyn_margin_y + h_v, dyn_margin_x:dyn_margin_x + w_v] = vis_dyn_view

        for detection in current_result.get("detections", []):
            x_m, y_m, w_m, h_m = detection["bbox"]
            dsme_result = detection["models"]["dsme"]
            color = (0, 255, 0)
            if dsme_result["label"] == 'Surprise':
                color = (0, 255, 255)
            elif dsme_result["label"] == 'Negative':
                color = (0, 0, 255)
            elif dsme_result["label"] == 'No Face':
                color = (150, 150, 150)

            line1 = f"{dsme_result['label']} {dsme_result['confidence']}%"

            text_y = max(18, y_m - 10)
            cv2.putText(frame, line1, (x_m, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


@app.route('/')
def index(): return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed(): return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_result')
def get_result(): return jsonify(current_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
