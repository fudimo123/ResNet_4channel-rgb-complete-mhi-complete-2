import os
import fusion_train as ft


def main():
    # 使用本目录下的融合特征图像与结果输出目录（fusion_train.py 已设置为本地路径）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fused_image_dir = os.path.join(current_dir, 'fusion_image')
    result_dir = os.path.join(current_dir, 'fusion_result1')
    # Sync path checks with fusion_train.py
    data_dir = os.path.normpath(os.path.join(current_dir, '..', 'data', 'CASME2_RAW_selected', 'CASME2_RAW_selected'))
    annotation_file = os.path.normpath(os.path.join(current_dir, '..', 'data', 'CASME2_RAW_selected', 'CASME2-coding-20140508.xlsx'))

    # 仅提示检查目录存在，训练入口直接调用本地 fusion_train.main()
    if not os.path.isdir(fused_image_dir):
        print(f"[WARN] 融合图像目录不存在: {fused_image_dir}")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    if not os.path.isdir(data_dir):
        print(f"[WARN] 数据目录不存在: {data_dir}，请按要求放置原始视频与标注文件")
    if not os.path.isfile(annotation_file):
        print(f"[WARN] 标注文件不存在: {annotation_file}，请按要求放置标注文件")

    # 运行训练（LOSO + 多次运行求均值/方差），结果会保存在当前目录 fusion_result1 下
    ft.main()


if __name__ == '__main__':
    main()