import os
import cv2
import torch
import numpy as np

from tools.Depth_Anything_V2.checkpoints.depth_anything_v2_vitl import DepthAnythingV2 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_configs = {
    'vitl': {
        'encoder': 'vitl',
        'features': 256,
        'out_channels': [256, 512, 1024, 1024]
    }
}

encoder = 'vitl'
model = DepthAnythingV2(**model_configs[encoder])
state_dict = torch.load("tools/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

input_folder = "results/temporary"
output_folder = "results/temporary_depth"
os.makedirs(output_folder, exist_ok=True)
# 遍历输入文件夹中的所有图片.后面指定输出的全景图
for file_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, file_name)
    image = cv2.imread(image_path)
    if image is None:
        continue
    with torch.no_grad():
        depth = model.infer_image(image)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255.0
    depth_norm = depth_norm.astype(np.uint8)

    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, depth_norm)