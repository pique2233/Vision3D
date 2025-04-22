import os
import sys
import torch
import yaml
from importlib import import_module

# 将项目根目录添加到 sys.path 中，确保可以导入 tools 模块
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

class MVD_Tools():
    def __init__(self, config_file: str = None, weight_path: str = None, device: str = 'cuda', outpaint: bool = False, strict: bool = True):
        """
        封装 MVDiffusion 模型接口，根据 outpaint 参数选择 PanoGenerator 或 PanoOutpaintGenerator
        """
        self.device = device
        self.strict = strict

        # 如果未提供配置文件路径，则使用默认路径（注意大小写）
        if config_file is None:
            config_file = 'tools/MVDiffusion/configs/pano_generation_outpaint.yaml' if outpaint else 'tools/MVDiffusion/configs/pano_generation.yaml'

        # 加载 YAML 配置文件
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件 {config_file} 未找到，请检查路径。")
        except yaml.YAMLError as e:
            raise ValueError(f"加载 YAML 配置文件时出错：{e}")

        # 动态导入生成器模块
        try:
            if outpaint:
                generator_module = import_module("tools.MVDiffusion.src.lightning_pano_outpaint")
                self.model = generator_module.PanoOutpaintGenerator(self.config)
            else:
                generator_module = import_module("tools.MVDiffusion.src.lightning_pano_gen")
                self.model = generator_module.PanoGenerator(self.config)
        except ModuleNotFoundError as e:
            raise ImportError(f"无法导入所需模块，请检查路径和依赖：{e}")

        # 加载权重文件
        if weight_path:
            try:
                checkpoint = torch.load(weight_path, map_location=device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                self.model.load_state_dict(state_dict, strict=self.strict)
            except FileNotFoundError:
                raise FileNotFoundError(f"权重文件 {weight_path} 未找到，请检查路径。")
            except Exception as e:
                raise RuntimeError(f"加载权重文件时发生错误：{e}")

        self.model.to(device)
        self.model.eval()

    def __call__(self, batch: dict):
        """
        对输入 batch 进行推理
        """
        # 确保数据在同一设备上
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            output = self.model.inference(batch)
        return output