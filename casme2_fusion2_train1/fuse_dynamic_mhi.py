import os
import re
import cv2
import numpy as np


def build_mhi_index(mhi_dir: str):
    """
    构建 mhi 文件索引：键为不含 onset/offset 的序列基名（如 EP03_02f），值为对应的 mhi 文件完整路径。
    """
    index = {}
    if not os.path.isdir(mhi_dir):
        return index
    for fname in os.listdir(mhi_dir):
        if not fname.lower().endswith('.png'):
            continue
        # 例如：EP03_02f_onset33_offset83_mhi.png -> 提取 EP03_02f
        m = re.match(r"^(EP[^_]+_\d+[^_]*)_onset.*_mhi\.png$", fname)
        if m:
            base = m.group(1)
            index[base] = os.path.join(mhi_dir, fname)
        else:
            # 兜底：截取到第一个 '_onset' 之前
            if '_onset' in fname:
                base = fname.split('_onset')[0]
                index[base] = os.path.join(mhi_dir, fname)
    return index


def fuse_images(dynamic_img: np.ndarray, mhi_img_gray: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    将动态成像（BGR）与 MHI（灰度）按加权平均融合为一张 BGR 图。
    alpha: 动态成像权重；(1-alpha): MHI 权重
    """
    h, w = dynamic_img.shape[:2]
    # 调整 mhi 尺寸以匹配动态图尺寸
    mhi_resized = cv2.resize(mhi_img_gray, (w, h), interpolation=cv2.INTER_LINEAR)
    # 灰度转 BGR 以便逐通道融合
    mhi_bgr = cv2.cvtColor(mhi_resized, cv2.COLOR_GRAY2BGR)

    # 转浮点融合
    dyn_f = dynamic_img.astype(np.float32)
    mhi_f = mhi_bgr.astype(np.float32)

    fused = cv2.addWeighted(dyn_f, alpha, mhi_f, 1.0 - alpha, 0.0)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    return fused


def main(alpha: float = 0.6):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dynamic_root = os.path.join(base_dir, 'dynamic_data')
    mhi_root = os.path.join(base_dir, 'casme2_mhi')
    fusion_root = os.path.join(base_dir, 'fusion_image')
    os.makedirs(fusion_root, exist_ok=True)

    total = 0
    fused_cnt = 0
    missing_mhi = 0
    error_cnt = 0

    # 遍历 subject 目录
    for sub_name in sorted(os.listdir(dynamic_root)):
        sub_dyn_dir = os.path.join(dynamic_root, sub_name)
        if not os.path.isdir(sub_dyn_dir):
            continue

        sub_mhi_dir = os.path.join(mhi_root, sub_name)
        if not os.path.isdir(sub_mhi_dir):
            print(f"[WARN] 缺少 MHI 目录: {sub_mhi_dir}")
            continue

        sub_fusion_dir = os.path.join(fusion_root, sub_name)
        os.makedirs(sub_fusion_dir, exist_ok=True)

        # 构建该 subject 的 mhi 索引
        mhi_index = build_mhi_index(sub_mhi_dir)

        for fname in sorted(os.listdir(sub_dyn_dir)):
            if not fname.lower().endswith('.jpg'):
                continue
            total += 1
            dyn_path = os.path.join(sub_dyn_dir, fname)
            # 从动态图文件名中提取基名：s{subject}_{base}.jpg -> {base}
            # 例如：s2_EP03_02f.jpg -> EP03_02f
            base = fname
            if '_' in base:
                base = base.split('_', 1)[1]  # 去掉 "s{subject}_"
            base = os.path.splitext(base)[0]  # 去掉 .jpg

            mhi_path = mhi_index.get(base)
            if not mhi_path or not os.path.isfile(mhi_path):
                missing_mhi += 1
                print(f"[MISS] {sub_name} 未找到对应 MHI: {base}")
                continue

            try:
                dyn_img = cv2.imread(dyn_path, cv2.IMREAD_COLOR)
                mhi_img = cv2.imread(mhi_path, cv2.IMREAD_GRAYSCALE)
                if dyn_img is None or mhi_img is None:
                    error_cnt += 1
                    print(f"[ERR] 读取图像失败: dyn={dyn_path}, mhi={mhi_path}")
                    continue

                fused = fuse_images(dyn_img, mhi_img, alpha=alpha)

                # 使用动态图的文件名保存，保持分类方式一致（按 subject 分目录，文件名仍以 s{subject}_ 开头）
                out_path = os.path.join(sub_fusion_dir, fname)
                ok = cv2.imwrite(out_path, fused)
                if ok:
                    fused_cnt += 1
                else:
                    error_cnt += 1
                    print(f"[ERR] 保存失败: {out_path}")
            except Exception as e:
                error_cnt += 1
                print(f"[ERR] 融合异常: {dyn_path} & {mhi_path} -> {e}")

    print("==== 完成融合 ====")
    print(f"总动态图: {total}")
    print(f"成功融合: {fused_cnt}")
    print(f"缺少 MHI: {missing_mhi}")
    print(f"错误数: {error_cnt}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fuse dynamic images and MHI by weighted averaging.')
    parser.add_argument('--alpha', type=float, default=0.6, help='Dynamic image weight (0-1). Default 0.6')
    args = parser.parse_args()
    main(alpha=args.alpha)