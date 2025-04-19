# 项目目录结构
```plaintext
Vision3D/   
├── config/ 配置文件
├── data/   数据集
├── ops/   操作文件夹，存放操作相关代码
├── pipe/    管道文件夹，用于组织多步骤的流程（如数据预处理管道、训练管道等）
├── README.md
├── results/   存放实验结果、日志、图表等输出文件
│   ├── temporary #存放临时文件
├── script/    # 辅助脚本如损失函数等
│   ├── loss_function
└── tools/ # 放置模型
    ├── Depth-Anything-V2/

```



# 全景图模块

# depth_anything_v2 模块的使用
```python    
python run.py \
  --encoder <vits | vitb | vitl | vitg> \
  --img-path <图片路径或图片目录，或包含图片路径的文本文件> \
  --outdir <输出文件夹> \
  [--input-size <尺寸>] [--pred-only] [--grayscale]
```
### 参数说明
- **--encoder**：选择模型的编码器架构，可选项有 vits、vitb、vitl 或 vitg。
- **vits**
encoder: 使用的小型 transformer 编码器，名称为 "vits"。
features: 中间特征的通道数为 64。
out_channels: 解码器各阶段输出的通道数依次为 48、96、192、384。
- **vitb**
encoder: 使用的基础 transformer 编码器，名称为 "vitb"。
features: 中间特征通道数为 128。
out_channels: 对应输出通道分别为 96、192、384、768。
- **vitl**
encoder: 使用的大型 transformer 编码器，名称为 "vitl"。
features: 中间特征通道数为 256。
out_channels: 各阶段输出的通道分别为 256、512、1024、1024。
- **vitg**
encoder: 使用的巨型 transformer 编码器，名称为 "vitg"。
features: 中间特征通道数为 384。
out_channels: 各阶段输出通道均为 1536（共四层）。
简单来说，这四个配置决定了模型采用的编码器类型、提取的特征数量以及解码器层次中各层输出的通道数，从而影响模型的性能和运行速度。
- **--img-path**：可以是单张图片路径、存放图片的文件夹，或者是包含多张图片路径的文本文件。
- **--outdir**：预测结果存放的输出目录。
- **--input-size**（可选）：指定输入图片尺寸，默认尺寸为 518，可调大以获取更细致的结果。
代码中通过调整默认输入尺寸
```python
    parser.add_argument('--input-size', type=int, default=518)
```
- **--pred-only**（可选）：只保存深度预测图，不附带原图。
- **--grayscale**（可选）：保存灰度深度图，不使用调色板渲染。
可以用尝试运行
```shéll
   python run.py --encoder vitl --img-path your_image.jpg --outdir output_dir --input-size 640
``` 

## 与其他模块整合
### 与全景图生成模块整合

# 修复模型模块


# 相机旋转策略

# 3D重建模块