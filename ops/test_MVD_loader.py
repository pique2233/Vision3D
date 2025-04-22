import os
import cv2
import yaml
import torch
import numpy as np
from PIL import Image

# 根据当前脚本位置计算项目根目录（Vision3D/）
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0,  1]], np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R

def resize_and_center_crop(img, size):
    H, W, _ = img.shape
    if H == W:
        img = cv2.resize(img, (size, size))
    elif H > W:
        current_size = int(size * H / W)
        img = cv2.resize(img, (size, current_size))
        margin = (current_size - size) // 2
        img = img[margin:margin+size, :]
    else:
        current_size = int(size * W / H)
        img = cv2.resize(img, (current_size, size))
        margin = (current_size - size) // 2
        img = img[:, margin:margin+size]
    return img

# 使用真实图像和提示词
image_path = os.path.join(base_dir, 'tools', 'MVDiffusion', 'assets', 'outpaint_example.png')
text_path = os.path.join(base_dir, 'tools', 'MVDiffusion', 'assets', 'prompts.txt')

# 加载 outpaint 模式的配置，用于真实图像验证
config_file = os.path.join(base_dir, 'tools', 'MVDiffusion', 'configs', 'pano_generation_outpaint.yaml')
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
resolution = config['dataset']['resolution']

# 读取图像并预处理
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"图像文件 {image_path} 未找到。")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = resize_and_center_crop(img, resolution)
img = img / 127.5 - 1  # 归一化到[-1, 1]
img = torch.tensor(img, dtype=torch.float32).cuda()

# 读取提示词文件，每行一个提示词，要求共8行；不足时复制第一行
with open(text_path, 'r') as f:
    prompt = [line.strip() for line in f if line.strip()]
if len(prompt) < 8:
    prompt = [prompt[0]] * 8

# 构造8个摄像头的内参和旋转矩阵（demo.py中以 90度视场，45度步长为例）
Ks, Rs = [], []
for i in range(8):
    degree = (45 * i) % 360
    K_val, R_val = get_K_R(90, degree, 0, resolution, resolution)
    Ks.append(K_val)
    Rs.append(R_val)
K_tensor = torch.tensor(Ks, dtype=torch.float32).cuda()[None]
R_tensor = torch.tensor(Rs, dtype=torch.float32).cuda()[None]

# 构造图像 batch，创建全0占位tensor，将第一视角替换为真实图像
B, N, H, W, C = 1, 8, resolution, resolution, 3
images = torch.zeros((B, N, H, W, C), dtype=torch.float32).cuda()
images[0, 0] = img

batch = {
    'images': images,
    'prompt': prompt,
    'K': K_tensor,
    'R': R_tensor
}

# 注意需要使用相同方式计算 weight_path
from MVD_loader import MVD_Tools
weight_path = os.path.join(base_dir, 'tools', 'MVDiffusion', 'weights', 'pano_outpaint.ckpt')
tool = MVD_Tools(config_file=config_file,
                 weight_path=weight_path,
                 device='cuda',
                 outpaint=True)
output = tool(batch)

# 指定输出目录（例如保存到 results/temporary）
output_dir = os.path.join(base_dir, 'results', 'temporary')
os.makedirs(output_dir, exist_ok=True)

# 假设 output 的 shape 为 (1, 8, H, W, 3)，逐视角保存图片
for i in range(8):
    image_array = output[0, i].cpu().numpy().astype('uint8')
    output_path = os.path.join(output_dir, f'output_{i}.png')
    im = Image.fromarray(image_array)
    im.save(output_path)
    print(f"已保存图像到 {output_path}")