
<div align="center">
  <h2>
    <a href="https://vistadream-project-page.github.io/" target="_blank" style="text-decoration:none; color:#007ACC;">Vision3D</a>
  </h2>
  <div>
    <a href="https://pique2233.github.io/Vision3D.github.io" target="_blank">
      <img src="https://img.shields.io/badge/Project%20Page-Click%20Here-blue?style=for-the-badge" alt="Project Page">
    </a>
  </div>
</div>

> **Vision3D**  
> [Ziwen Li](https://pique2233.github.io/ziwenli.github.io/), [XianFeng Han](), [Guanyu Qv]()  
> [**Project-page (with Interactive DEMOs)**](https://pique2233.github.io/Vision3D.github.io//)

---

## 🔭 Introduction

**Vision3D** 是一个用于 3D 场景重建和深度估计的开源项目，结合了多种先进的深度学习技术。它支持从单张图像生成深度图、全景图生成以及 3D 场景建模等功能。

Vision3D 接收单张图像（或全景图作为输入），首先利用基于 Stable Diffusion 和 ControlNet Inpainting 的预训练模型，配合自设的 prompt 模板进行图像扩展和完善，生成 2:1 比例的全景图。后续将全景图切分为 8-14 张图片，分别进行精密深度估计，采用 Depth Anything V2 进行 metric 深度估计，以获得高质量的深度图。

然后，将图像和深度图联合输入 Nerfstudio 应用中的 Gaussian Splatting (GS) 模块，无需训练即可实现高效的 3D 重建效果。

在初步重建基础上，我们使用基于重建场景的视角控制扩散模型（如 Zero-1-to-3 或 View-Conditioned Diffusion）合成新视图，用于输入 GS 模块进行增量重建，补全遮挡区域，优化空间连续性。

同时，支持 Mesh 网格导出功能，可与 NeRF-SLAM 或 instant-ngp 等环境兼容，方便实时应用。

最终，**Vision3D** 打通了从单图输入到可交互 3D 场景输出的完整链路，具备良好的扩展性与实际应用前景。

---

## 🆕 News

- **2023-10-01**: 发布了 `Depth-Anything-V2` 模块的集成，支持更高精度的深度估计；
- **2023-09-15**: 增加了全景图生成模块，支持多视角拼接；
- **2023-08-30**: 项目正式开源，欢迎贡献！

---

## 💻 Requirements

- Python >= 3.9
- CUDA >= 11.7 (for GS-based rendering)
- 其他依赖请参考各模块的 `requirements.txt` 文件

安装依赖：

```bash
pip install -r requirements.txt
```

---

## ✅ To Do List


- [ ] **深度估计与三维重建流程自动化**：打通 Depth Anything → GS 重建全流程  
- [ ] **支持 Mesh 输出与多格式切换**：添加重建结果的 mesh 导出与预览切换（.ply / .glb）  
- [ ] **发布技术说明文档**：完善系统各模块的使用手册与原理说明，提升工程复现性  
- [ ] **增加更多的 Demo 案例**：补充多场景、多角度的样例数据，展示模型泛化能力 



---

## 🔗 Acknowledgement

This work is built on many amazing open source projects, thanks to all the authors!

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) for the wonderful monocular metric depth estimation accuracy  
- [StableDiffusion](https://github.com/CompVis/stable-diffusion) for the wonderful image generation/optimization ability  
- [ControlNet](https://github.com/lllyasviel/ControlNet) for controllable condition-driven generation  
- [MiDaS](https://github.com/isl-org/MiDaS) for additional baseline depth estimation  
- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) for flexible NeRF pipeline and GS implementation  
- [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) for view synthesis module  
- [WonderGen](https://github.com/ZiYang-xie/WorldGen) for the World Gen

---

## 🔖 BibTeX

```bibtex
@misc{li2025vision3d,
  author       = {Ziwen Li and Xianfeng Han},
  title        = {Vision3D: Panoramic 3D Scene Reconstruction from a Single Image via Diffusion and Gaussian Splatting},
  year         = {2025},
  howpublished = {GitHub},
  url          = {https://pique2233.github.io/Vision3D.github.io}
}
```

---
