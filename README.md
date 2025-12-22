# info_sEEG
Information seeking_SEEG study led by Keyu



### 简介 / Introduction

本项目探索人类前额叶皮层（尤其是vmPFC和ACC）在动态调节奖励和信息价值中的神经机制。通过结合计算模型、腔内电生理（SEEG）数据和网络分析，本研究揭示了大脑在探索-利用权衡过程中的不同神经编码和连接状态的切换。这些结果为理解人类灵活决策提供了神经基础和机制解释。

**Research highlights include**
- 识别前额叶不同区域对奖励和信息价值的不同编码方式
- Theta频段的长程同步与价值整合的关系
- 负责任的局部及区域间的环路动态调控探究

### 文件结构 / Folder Structure

```plaintext
/ --> 代码仓库根目录
│
├── data/                # 样本数据（如有存档，非原始患者隐私数据）
│
├── analysis/            # 主要分析脚本和流程
│   ├── models/          # 计算模型（gkRL等）实现
│   ├── sEEG/          # SEEG信号处理与分析脚本
│   ├── connectivity/    # 网络连接与频段分析（PLV, PSI等）
│   └── visualization/   # 图表和结果可视化脚本
│
├── results/             # 输出的分析结果和图像
│
├── docs/                # 研究相关文档、方法说明、依赖说明
│
└── README.md            # 本说明文件

### How to Use

- **Code and scripts** are designed to reproduce key analyses such as neural encoding, connectivity measures, and modeling.
- **Data** files (when available) are for internal use; user-specific data should be preprocessed accordingly.
- For questions, bug reports, or collaboration requests, please contact us at `keyuhu@um.edu.mo`or`haiyanwu@um.edu.mo`.

About / 关于

This repository contains code and analysis scripts for the study of neural mechanisms underlying flexible reward and information processing in human prefrontal circuits.
本仓库收录了支持本研究的核心代码和分析流程，可作补充和方法复现。

License / 使用协议

此研究由 ANDlab完成，截图和数据（限非敏感部分）可用于学术交流与再研究。敬请注明出处。

**Thank you for your interest!**
感谢您的关注！
