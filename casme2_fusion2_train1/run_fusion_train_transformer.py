import os
import fusion_train as ft
from fusion_model_spp_transformer import create_fusion_model as create_transformer_model


def main():
    # 当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fused_image_dir = os.path.join(current_dir, 'fusion_image')
    result_dir = os.path.join(current_dir, 'fusion_result3')
    # 与 fusion_train.py 同步的数据路径检查
    data_dir = os.path.normpath(os.path.join(current_dir, '..', 'data', 'CASME2_RAW_selected', 'CASME2_RAW_selected'))
    annotation_file = os.path.normpath(os.path.join(current_dir, '..', 'data', 'CASME2_RAW_selected', 'CASME2-coding-20140508.xlsx'))

    # 路径存在性提示
    if not os.path.isdir(fused_image_dir):
        print(f"[WARN] 融合图像目录不存在: {fused_image_dir}")
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    if not os.path.isdir(data_dir):
        print(f"[WARN] 数据目录不存在: {data_dir}，请按要求放置原始视频与标注文件")
    if not os.path.isfile(annotation_file):
        print(f"[WARN] 标注文件不存在: {annotation_file}，请按要求放置标注文件")

    # 覆盖模型工厂为 Transformer 版本，并将结果目录改为 fusion_result3
    ft.create_fusion_model = create_transformer_model
    ft.RESULT_DIR = 'fusion_result3'

    # 运行训练（LOSO），结果会保存在当前目录 fusion_result3 下
    ft.main()


if __name__ == '__main__':
    main()