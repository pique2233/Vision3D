# language: python
import os
import cv2
import torch
import numpy as np

# 从 Depth Anything V2 模型库中导入 DepthAnythingV2 类
from tools.Depth_Anything_V2.checkpoints.depth_anything_v2_vitl import DepthAnythingV2  # 请根据实际的模型导出文件调整导入路径

# 设置设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型配置（可参考 [tools/Depth-Anything-V2/app.py](tools/Depth-Anything-V2/app.py) 中的配置）
model_configs = {
    'vitl': {
        'encoder': 'vitl',
        'features': 256,
        'out_channels': [256, 512, 1024, 1024]
    }
}

encoder = 'vitl'
# 加载模型及权重
model = DepthAnythingV2(**model_configs[encoder])
state_dict = torch.load("tools/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

input_folder = "results/temporary"
output_folder = "results/temporary_depth"
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹里所有图像进行深度预测
for file_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, file_name)
    image = cv2.imread(image_path)
    if image is None:
        continue

    # 推理：此处假设模型有 infer_image 方法，可参考 [tools/Depth-Anything-V2/app.py](tools/Depth-Anything-V2/app.py)
    with torch.no_grad():
        depth = model.infer_image(image)

    # 将深度图归一化并转换为 8 位图
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255.0
    depth_norm = depth_norm.astype(np.uint8)

    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, depth_norm)