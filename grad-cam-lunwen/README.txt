grad-cam-lunwen

用途：
1. 使用 CASME2 旧版 ADF-Net 权重生成论文可视化热力图。
2. 同时输出单分支热力图和双分支对照图。

当前测试权重：
casme2_fusion_train2_3class copy\fusion_result_asym_se-1\fusion_resnet18_casme2_final_model_for_gui.pth

主脚本：
generate_adfnet_gradcam_samples.py

默认行为：
1. 从 CASME2 中自动挑选 positive / negative / surprise 各 1 个样本。
2. 对 RGB 分支生成 Grad-CAM 和 Grad-CAM++。
3. 对 Dynamic 分支生成 Grad-CAM 和 Grad-CAM++。
4. 生成双分支对照图。
5. 所有结果保存到 outputs\ 下。

常用命令：
python generate_adfnet_gradcam_samples.py

若以后切换成新的权重，可直接指定：
python generate_adfnet_gradcam_samples.py --checkpoint "你的新权重路径"

若只想生成 Grad-CAM++：
python generate_adfnet_gradcam_samples.py --methods gradcampp

可切换 target layer：
python generate_adfnet_gradcam_samples.py --methods gradcampp --rgb-target layer4 --dynamic-target layer3

当前常用 target layer：
- RGB branch: cbam / layer4 / layer3
- Dynamic branch: layer4 / layer3 / layer2
