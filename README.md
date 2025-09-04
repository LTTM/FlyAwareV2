# 🚁 FlyAwareV2: Multi-Modal UAV Dataset for Urban Semantic Segmentation

[![License: GPL3](https://img.shields.io/badge/License-GPL3-yellow.svg)](https://opensource.org/license/gpl-3-0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org)
[![Dataset](https://img.shields.io/badge/Dataset-Official%20Website-green)](https://flyaware-dataset.org)

A comprehensive multi-modal dataset for semantic segmentation in urban UAV scenarios, featuring both synthetic and real data across different weather conditions, altitudes, and modalities.

---

## 👥 Authors

<div align="center">

| [Francesco Barbato*](https://medialab.dei.unipd.it/members/francesco-barbato/) | [Matteo Caligiuri*](https://matteocali.github.io/) | [Pietro Zanuttigh](https://medialab.dei.unipd.it/members/pietro-zanuttigh/) |
|:---:|:---:|:---:|

**Department of Information Engineering, University of Padova**,
Via Gradenigo 6/b, 35131 Padova, Italy

\* These authors contributed equally to this work.
</div>

---

## 📊 Graphical Abstract

<div align="center">

<img src="extras/graphabstract.svg" alt="FlyAwareV2 Graphical Abstract" width="60%">

</div>

---

## 📖 Overview

**FlyAwareV2** is a large-scale multi-modal dataset designed for semantic segmentation of urban aerial imagery captured by UAVs (Unmanned Aerial Vehicles). The dataset addresses the critical need for robust computer vision models in adverse weather conditions and diverse urban environments.

### 🌟 Key Features

- 🏙️ **Multi-Environment**: Multiple urban towns and scenarios
- 📏 **Multi-Altitude**: Different recording heights (20m, 50m, 120m)
- 🎯 **Multi-Modal**: RGB, Depth, and Semantic annotations
- 🌦️ **Adverse Weather**: Sunny, Rainy, Foggy, and Night conditions
- 🔄 **Synthetic + Real**: CARLA-generated synthetic data + augmented real imagery
- 📊 **Comprehensive Benchmarks**: Complete evaluation suite with domain adaptation

> [!IMPORTANT]
> This dataset is specifically designed for **adverse weather analysis** in urban UAV scenarios, making it unique for studying weather-robust semantic segmentation algorithms.

---

## 🎯 Citation

If you use FlyAwareV2 in your research, please cite our paper:

```bibtex
@article{flyawarev2_2025,
  title={FlyAwareV2: Multi-Modal UAV Dataset for Urban Semantic Segmentation with Adverse Weather Analysis},
  author={Francesco Barbato and Matteo Caligiuri and Pietro Zanuttigh},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  institution={Dipartimento di Ingegneria dell'Informazione (DEI), University of Padova}
}
```

---

## ⬇️ Dataset Download

> [!IMPORTANT]
> **The FlyAwareV2 dataset is now available for download!** Choose the version that best fits your research needs.

### 🔗 Download Links

| Dataset Version | Size | Description | Download |
|----------------|------|-------------|----------|
| 🎮 **Synthetic Only** | ~XX GB | CARLA-generated data with all weather conditions | [Download Synthetic](https://flyaware-dataset.org/download/synthetic) |
| 📷 **Real Only** | ~XX GB | Augmented real UAV imagery from UAVid & VisDrone | [Download Real](https://flyaware-dataset.org/download/real) |
| 🔄 **Complete Dataset** | ~XX GB | Both synthetic and real data (Recommended) | [Download Complete](https://flyaware-dataset.org/download/complete) |

### 📁 Recommended Folder Structure

After downloading and extracting the dataset, organize your data following this structure:

```text
FlyAwareV2/
├── 📁 real/
│   ├── 📁 train/
│   │   ├── 📁 day/                     # Clear weather training data
│   │   │   ├── 📁 rgb/                 # RGB images
│   │   │   └── 📁 depth/               # Depth maps
│   │   ├── 📁 fog/                     # Foggy training data
│   │   ├── 📁 night/                   # Night training data
│   │   └── 📁 rain/                    # Rainy training data
│   └── 📁 test/
│       ├── 📁 day/                     # Test data with annotations
│       │   ├── 📁 rgb/
│       │   ├── 📁 depth/
│       │   ├── 📁 semantic/            # Semantic segmentation
│       ├── 📁 fog/
│       ├── 📁 night/
│       └── 📁 rain/
└── 📁 synthetic/
    ├── 📁 Town01_Opt_120/              # Urban environment 1
    │   ├── 📁 ClearNoon/               # Sunny conditions
    │   │   ├── 📁 height20m/           # 20m altitude
    │   │   │   ├── 📁 rgb/
    │   │   │   ├── 📁 depth/
    │   │   │   ├── 📁 semantic/
    │   │   │   └── 📁 camera/          # Camera parameters
    │   │   ├── 📁 height50m/           # 50m altitude
    │   │   └── 📁 height80m/           # 80m altitude
    │   ├── 📁 HardRainNoon/            # Rainy conditions
    │   ├── 📁 MidFoggyNoon/            # Foggy conditions
    │   └── 📁 ClearNight/              # Night conditions
    ├── 📁 Town02_Opt_120/              # Additional towns...
    ├── 📁 Town03_Opt_120/
    ├── 📁 Town04_Opt_120/
    ├── 📁 Town05_Opt_120/
    ├── 📁 Town06_Opt_120/
    ├── 📁 Town07_Opt_120/
    └── 📁 Town10HD_Opt_120/
```

> [!NOTE]
> The complete folder structure contains over 100K+ images across all modalities and conditions. Each town includes 4 weather conditions and 3 altitude levels with RGB, depth, and semantic data.

---

## 📊 Dataset Statistics

| **Modality** | **Weather Conditions** | **Towns** | **Altitudes** | **Total Samples** |
|-------------|------------------------|-----------|---------------|-------------------|
| RGB + Depth + Semantic | Sunny, Rainy, Foggy, Night | 8 Towns | 3 Heights | 100K+ |

### 🌤️ Weather Conditions

| Weather | Description | Real Data | Synthetic Data |
|---------|-------------|-----------|----------------|
| ☀️ **Sunny** | Clear weather conditions | Native | Simulated |
| 🌧️ **Rainy** | Rain effects and wet surfaces | Augmented | Simulated |
| 🌫️ **Foggy** | Fog simulation with depth-aware effects | Augmented | Simulated |
| 🌙 **Night** | Low-light and artificial lighting | Augmented | Simulated |

---

## 🗂️ Repository Structure

This repository contains all the code and tools for dataset generation, processing, and evaluation:

```text
FlyAwareV2/
├── 📁 synthetic_data_generation/    # CARLA-based synthetic data generation
├── 📁 real_data_processing/         # Real data augmentation and processing
│   ├── 📁 fog/                     # Fog simulation tools
│   └── 📁 rain_and_night/          # Rain and night augmentation
├── 📁 benchmarks/                  # Comprehensive evaluation suite
├── 📁 extras/                      # Additional resources and assets
└── 📄 README.md                    # This file
```

### 🛠️ Components Overview

| Component | Purpose | Key Technologies |
|-----------|---------|------------------|
| **Synthetic Generation** | Generate realistic UAV imagery | Modified CARLA Simulator |
| **Real Data Processing** | Augment real imagery with weather effects | MonoFog, img2img-turbo |
| **Benchmarks** | Model evaluation and comparison | PyTorch, Domain Adaptation |

---

## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended)

### 1️⃣ Clone the Repository

```bash
git clone --recursive https://github.com/LTTM/FlyAwareV2.git
cd FlyAwareV2
```

### 2️⃣ Dataset Download

> [!NOTE]
> The complete dataset download instructions and links will be available on our [official website](https://flyaware-dataset.org).

---

## 🔧 Usage

### 🎮 Synthetic Data Generation

Generate synthetic UAV data using our modified CARLA simulator:

```bash
cd synthetic_data_generation
# Follow detailed instructions in synthetic_data_generation/README.md
python run_simulation.py --config configs/urban_config.yaml
```

**Key Features:**

- 🏙️ Multiple urban environments (8 towns)
- 🌦️ All weather conditions simulation
- 📐 Configurable flight altitudes
- 🎯 Automatic semantic annotation

### 🌊 Real Data Augmentation

Transform clear real images into adverse weather conditions:

#### 🌫️ Fog Generation

```bash
cd real_data_processing/fog
python clear2fog.py --input <path_to_images> --output <output_path>
```

#### 🌧️ Rain & Night Augmentation

```bash
cd real_data_processing/rain_and_night
python gradio_app.py  # Interactive interface
```

### 📈 Benchmarking & Evaluation

Comprehensive model evaluation with our benchmark suite:

```bash
cd benchmarks
# Pre-training on synthetic data
python synthetic_pretrain.py --root_path <dataset_path> --config <config_file>

# Evaluation on real data
python evaluate.py --root_path <dataset_path> --model_path <checkpoint_path>

# Domain adaptation
python UDA_finetune.py --source synthetic --target real
```

> [!TIP]
> Check the individual README files in each directory for detailed usage instructions and configuration options.

---

## 🎯 Applications

FlyAwareV2 is designed for various computer vision tasks:

- **🔍 Semantic Segmentation**: Urban scene understanding from aerial perspectives
- **🌦️ Adverse Weather Analysis**: Robust perception in challenging conditions
- **🔄 Domain Adaptation**: Bridging synthetic-to-real domain gaps
- **🚁 UAV Navigation**: Autonomous drone navigation in urban environments
- **📊 Benchmark Studies**: Standardized evaluation of aerial perception models

---

## 📚 Dataset Origins & Augmentation

### 🎮 Synthetic Data

- **Source**: Modified [CARLA Simulator](https://carla.org/)
- **Enhancement**: Custom urban scenarios and weather simulation
- **Coverage**: 8 different towns with varied architectural styles

### 📷 Real Data

- **Base Datasets**:
  - [UAVid](https://uavid.nl/) - Urban aerial imagery
  - [VisDrone](http://aiskyeye.com/) - Drone-based object detection dataset
- **Augmentation**: Custom weather transformation pipeline
- **Consistency**: Domain-aware augmentation preserving semantic coherence

---

## 🏆 Benchmarks & Results

Our benchmark suite evaluates models across multiple dimensions:

- **🎯 Semantic Segmentation Performance**: mIoU, accuracy metrics
- **🌦️ Weather Robustness**: Performance degradation analysis
- **🔄 Domain Adaptation**: Synthetic-to-real transfer learning
- **⚡ Computational Efficiency**: FLOPs and inference time analysis

> [!NOTE]
> Detailed benchmark results and leaderboards are available in the [`benchmarks/`](benchmarks/) directory.

---

## 🤝 Contributing

We welcome contributions to improve FlyAwareV2! Please see our contributing guidelines:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. 💻 Make your changes
4. 🧪 Add tests if applicable
5. 📝 Update documentation
6. 🚀 Submit a pull request

---

## 📄 License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [CARLA Simulator](https://carla.org/) for the base simulation environment;
- [UAVid Dataset](https://uavid.nl/) and [VisDrone Dataset](http://aiskyeye.com/) for real aerial imagery;
- [imag2img-turbo](https://github.com/JimmyxGuo/-img2img-turbo) and [FoHIS](https://github.com/noahzn/FoHIS/tree/master) for image translation tasks;
- [marigold](https://github.com/prs-eth/Marigold) for depth estimation;
- This work was partially supported by the European Union under the Italian National Recovery and Resilience Plan (NRRP) of NextGenerationEU, partnership on "Telecommunications of the Future" (PE00000001- program "RESTART").

---

## 📞 Support

For questions and support:

- 📧 Email: [francesco.barbato@unipd.it](mailto:francesco.barbato@unipd.it)
- 🐛 Issues: [GitHub Issues](https://github.com/LTTM/FlyAware/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/LTTM/FlyAware/discussions)

---

🚁 **Advancing UAV Perception in Urban Environments** 🏙️

Built with ❤️ by the [MEDIALab](https://medialab.dei.unipd.it/) Research Group
