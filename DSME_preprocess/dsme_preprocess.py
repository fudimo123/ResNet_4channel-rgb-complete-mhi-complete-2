import os
import sys
import math
import shutil
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN

def list_videos(input_dir):
    items = []
    if not os.path.isdir(input_dir):
        return items
    children = [d for d in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, d))]
    emotion_names = {"disgust", "fear", "happiness", "sadness", "surprise", "digust", "hapiness", "suprise"}
    if any(c.lower() in emotion_names for c in children):
        person = os.path.basename(input_dir.rstrip(os.sep))
        for emotion in children:
            epath = os.path.join(input_dir, emotion)
            if not os.path.isdir(epath):
                continue
            for video in sorted(os.listdir(epath)):
                vpath = os.path.join(epath, video)
                if os.path.isfile(vpath) and vpath.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    items.append((person, emotion, video, vpath))
        return items
    for person in children:
        ppath = os.path.join(input_dir, person)
        for emotion in sorted(os.listdir(ppath)):
            epath = os.path.join(ppath, emotion)
            if not os.path.isdir(epath):
                continue
            for video in sorted(os.listdir(epath)):
                vpath = os.path.join(epath, video)
                if os.path.isfile(vpath) and vpath.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    items.append((person, emotion, video, vpath))
    return items

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def extract_face_roi(img_bgr, mtcnn, margin=20):
    if mtcnn is None:
        return img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(Image.fromarray(img_rgb))
    if boxes is None or len(boxes) == 0:
        return img_bgr
    box = boxes[0]
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(box[0]) - margin)
    y1 = max(0, int(box[1]) - margin)
    x2 = min(w, int(box[2]) + margin)
    y2 = min(h, int(box[3]) + margin)
    roi = img_bgr[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return img_bgr
    return roi

def save_frames(frames, out_dir):
    ensure_dir(out_dir)
    for x in os.listdir(out_dir):
        if x.startswith('img') and x.endswith('.jpg'):
            try:
                os.remove(os.path.join(out_dir, x))
            except Exception:
                pass
    idx = 1
    for f in frames:
        cv2.imwrite(os.path.join(out_dir, f"img{idx}.jpg"), f)
        idx += 1

def interpolate_with_ffmpeg(video_path, target_fps, out_dir):
    ensure_dir(out_dir)
    tmp_pattern = os.path.join(out_dir, "raw_%05d.jpg")
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vf', f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,format=yuv420p",
        '-vsync', '0', tmp_pattern
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    raws = sorted([os.path.join(out_dir, x) for x in os.listdir(out_dir) if x.startswith('raw_') and x.endswith('.jpg')])
    frames = []
    for rp in raws:
        img = cv2.imread(rp)
        if img is None:
            continue
        frames.append(img)
    for rp in raws:
        try:
            os.remove(rp)
        except Exception:
            pass
    if len(frames) == 0:
        return False
    save_frames(frames, out_dir)
    return True

def interpolate_frames_opencv(frames_bgr, orig_fps, target_fps):
    if len(frames_bgr) == 0:
        return []
    duration = (len(frames_bgr) - 1) / float(orig_fps)
    target_count = int(math.floor(duration * target_fps)) + 1
    if target_count <= 1:
        return [frames_bgr[0]]
    res = []
    for j in range(target_count):
        t = j / float(target_fps)
        s = t * orig_fps
        i = int(math.floor(s))
        if i >= len(frames_bgr) - 1:
            res.append(frames_bgr[-1])
            continue
        alpha = s - i
        f0 = frames_bgr[i]
        f1 = frames_bgr[i + 1]
        blended = cv2.addWeighted(f0, 1.0 - alpha, f1, alpha, 0)
        res.append(blended)
    return res

def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def _limit_frames(frames, max_frames):
    if max_frames is None or max_frames <= 0:
        return frames
    if len(frames) <= max_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int)
    return [frames[i] for i in idx]

def process_video(video_path, out_dir, target_fps, mtcnn, max_frames=None):
    print(f"processing {video_path} -> {out_dir}")
    ffmpeg_ok = shutil.which('ffmpeg') is not None
    ensure_dir(out_dir)
    if ffmpeg_ok:
        print("using ffmpeg")
        ok = interpolate_with_ffmpeg(video_path, target_fps, out_dir)
        if ok:
            final_frames = []
            imgs = sorted([os.path.join(out_dir, x) for x in os.listdir(out_dir) if x.startswith('img') and x.endswith('.jpg')])
            for ip in imgs:
                img = cv2.imread(ip)
                if img is None:
                    continue
                roi = extract_face_roi(img, mtcnn)
                final_frames.append(roi)
            final_frames = _limit_frames(final_frames, max_frames)
            for ip in imgs:
                try:
                    os.remove(ip)
                except Exception:
                    pass
            save_frames(final_frames, out_dir)
            print("ffmpeg done")
            return True
    raw_frames, orig_fps = read_video_frames(video_path)
    print(f"opencv read frames: {len(raw_frames)} fps={orig_fps}")
    if len(raw_frames) == 0:
        return False
    if orig_fps <= 0:
        orig_fps = 25.0
    interp_frames = interpolate_frames_opencv(raw_frames, orig_fps, target_fps)
    print(f"interpolated frames: {len(interp_frames)}")
    final_frames = []
    for f in interp_frames:
        try:
            roi = extract_face_roi(f, mtcnn)
        except Exception:
            roi = f
        final_frames.append(roi)
    final_frames = _limit_frames(final_frames, max_frames)
    try:
        print(f"to save frames: {len(final_frames)}")
        save_frames(final_frames, out_dir)
    except Exception as e:
        print(f"save error: {e}")
    print(f"saved {len(final_frames)} frames to {out_dir}")
    print("opencv done")
    return True

def process_dataset(input_dir, output_dir, target_fps=75, max_frames=60, apply_face_align=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=False, select_largest=True, device=device) if apply_face_align else None
    items = list_videos(input_dir)
    print(f"found {len(items)} videos")
    for person, emotion, video, vpath in items:
        vname = os.path.splitext(video)[0]
        out_dir = os.path.join(output_dir, person, emotion, vname)
        try:
            ok = process_video(vpath, out_dir, target_fps, mtcnn, max_frames=max_frames)
            print(f"ok={ok}")
            if not ok:
                continue
        except Exception as e:
            print(f"error processing {vpath}: {e}")
            continue

def main():
    base = Path(__file__).resolve().parents[1]
    input_dir = str(base / 'data' / 'DSME_video')
    output_dir = str(base / 'data' / 'DSME_pic')
    target_fps = 75
    max_frames = 60
    apply_face_align = False
    if len(sys.argv) >= 2:
        input_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    if len(sys.argv) >= 4:
        try:
            target_fps = int(sys.argv[3])
        except Exception:
            target_fps = 100
    if len(sys.argv) >= 5:
        try:
            max_frames = int(sys.argv[4])
        except Exception:
            max_frames = None
    if len(sys.argv) >= 6:
        try:
            apply_face_align = bool(int(sys.argv[5]))
        except Exception:
            apply_face_align = False
    ensure_dir(output_dir)
    process_dataset(input_dir, output_dir, target_fps, max_frames=max_frames, apply_face_align=apply_face_align)

def align_dataset(output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)
    for person in sorted(os.listdir(output_dir)):
        ppath = os.path.join(output_dir, person)
        if not os.path.isdir(ppath):
            continue
        for emotion in sorted(os.listdir(ppath)):
            epath = os.path.join(ppath, emotion)
            if not os.path.isdir(epath):
                continue
            for vname in sorted(os.listdir(epath)):
                vdir = os.path.join(epath, vname)
                if not os.path.isdir(vdir):
                    continue
                imgs = sorted([x for x in os.listdir(vdir) if x.startswith('img') and x.endswith('.jpg')])
                for img_name in imgs:
                    ip = os.path.join(vdir, img_name)
                    img = cv2.imread(ip)
                    if img is None:
                        continue
                    try:
                        roi = extract_face_roi(img, mtcnn)
                    except Exception:
                        roi = img
                    cv2.imwrite(ip, roi)

if __name__ == '__main__':
    main()