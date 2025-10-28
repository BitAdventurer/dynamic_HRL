# 🧠 Hierarchical Reinforcement Learning for MDD/NC Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-blue.svg)](LICENSE)

### Overview
<p align="center">
    <img src="/intro_fig.png" alt="drawing" width="800"/>
</p>

This project implements a **Hierarchical Reinforcement Learning (HRL)** system for classifying Major Depressive Disorder (MDD) and Normal Control (NC) subjects.

**Key Features:**
- 🎯 **Dynamic Window Selection**: Macro-Agent selects optimal window sizes
- 🔄 **Adaptive Step Adjustment**: Micro-Agent selects optimal shift ratios
- 🤖 **Evolution Algorithm**: Automated hyperparameter optimization
- 📊 **Multiple Classifiers**: CRNN, CBGRU, Transformer support
- 🔬 **Ablation Study**: Comprehensive component contribution analysis

## 📁 Project Structure

```
JBHI_season2/
├── main/                          # 🎯 Core source code
│   ├── train.py                   # HRL model training and evaluation script
│   ├── HRL.py                     # Hierarchical RL implementation (Macro/Micro Agents)
│   ├── EVO.py                     # Evolution algorithm for hyperparameter optimization
│   ├── models.py                  # Neural network definitions (CRNN, CBGRU, Transformer, Q-Networks)
│   ├── plot.py                    # Basic visualization (window/step distribution, learning curves)
│   ├── plot_ablation.py           # Ablation study result visualization
│   ├── plot_reward.py             # Reward analysis visualization
│   ├── plot_full_distribution.py  # Full distribution visualization
│   ├── plot_gradient.py           # Gradient analysis visualization
│   ├── plot_policy_log.py         # Policy learning visualization
│   ├── save_results.py            # Results saving utilities
│   ├── convert_models_to_state_dict.py  # Model conversion utilities
│   └── heatmap.py                 # Heatmap generation utilities
│
├── utils/                         # 🛠️ Common utilities
│   ├── preprocessing.py           # Data preprocessing (PCA, LDA, Z-score)
│   ├── seed.py                    # Seed setting for reproducibility
│   ├── util.py                    # Data loading and other utilities
│   └── __init__.py                # Utilities package initialization
│
├── config.py                      # ⚙️ Configuration and hyperparameters
├── requirements.txt               # 📦 Python dependencies
├── run_ablation.sh                # 🚀 Ablation study execution script
├── LICENSE                        # 📄 CC BY-NC-SA 4.0 International License
└── README.md                      # 📖 Project documentation
```

## 🚀 Quick Start

### 📋 Installation
```bash
# Clone the repository
git clone https://github.com/BitAdventurer/dynamic_HRL.git
cd dynamic_HRL

# Install dependencies
pip install -r requirements.txt
```

### 🎯 Basic Usage
```bash
# Basic HRL training with CRNN
python main/train.py

# Training with different classifiers
python main/train.py --classifier_type cbgru
python main/train.py --classifier_type transformer

# Run ablation study
bash run_ablation.sh
```

### 📊 Visualization
```bash
# Plot reward analysis
python main/plot_reward.py

# Plot full distribution
python main/plot_full_distribution.py

# Plot ablation results
python main/plot_ablation.py
```

## 🚀 Detailed Usage

### 1️⃣ HRL Model Training and Evaluation

**Basic Training (CRNN classifier):**
```bash
python main/train.py
```

**Using Different Classifiers:**
```bash
# CBGRU (Bidirectional GRU)
python main/train.py --classifier_type cbgru

# Spatio-Temporal Transformer
python main/train.py --classifier_type transformer \
    --transformer_num_heads 8 \
    --transformer_num_layers 4
```

**Inference Mode (evaluation only):**
```bash
python main/train.py --inference_only --model_fold 1
```

**Training Specific Fold Only:**
```bash
python main/train.py --run_fold 0  # Run Fold 1 only (0-indexed)
```

### 2️⃣ Evolution Algorithm for Hyperparameter Optimization

**HRL Hyperparameter Optimization:**
```python
# Set USE_HRL = True in main/EVO.py
python main/EVO.py
```

**Baseline Classifier Optimization:**
```python
# Set USE_HRL = False in main/EVO.py
python main/EVO.py
```

**Evolution Algorithm Settings:**
- `pop_size`: Population size (default: 4)
- `generations`: Number of generations (default: 3)
- `epochs`: Training epochs per individual (default: 10 for HRL, 20 for baseline)

**Optimized Hyperparameters (HRL mode):**
- Classifier: hidden_dim, num_layers, dropout, lr, wd
- Macro-Agent: lr, wd
- Micro-Agent: lr, wd
- RL: gamma, batch_size, buffer_capacity, target_update_freq
- Focal Loss: alpha, gamma

### 3️⃣ Ablation Study Execution

**Automated Script Execution:**
```bash
bash run_ablation.sh
```

**Individual Ablation Mode Execution:**
```bash
# Full HRL (default model)
python main/train.py --ablation_mode full

# Remove Macro Agent (fixed window size)
python main/train.py --ablation_mode no_macro --fixed_window_size 30

# Remove Micro Agent (fixed shift ratio)
python main/train.py --ablation_mode no_micro --fixed_shift_ratio 0.5

# Remove both Agents (completely fixed)
python main/train.py --ablation_mode fixed \
    --fixed_window_size 30 \
    --fixed_shift_ratio 0.5
```

**Ablation Mode Description:**
| Mode | Macro-Agent | Micro-Agent | Description |
|------|-------------|-------------|-------------|
| `full` | ✅ | ✅ | Complete HRL (dynamic window & step) |
| `no_macro` | ❌ | ✅ | Fixed window, dynamic step |
| `no_micro` | ✅ | ❌ | Dynamic window, fixed step |
| `fixed` | ❌ | ❌ | Fixed window & step (baseline) |

### 4️⃣ Visualization Execution

**Basic Visualization (learning curves, distributions):**
```bash
python main/plot.py
```

**Ablation Study Result Comparison:**
```bash
python main/plot_ablation.py
```

**Reward Analysis:**
```bash
python main/plot_reward.py
```

**Full Distribution Visualization:**
```bash
python main/plot_full_distribution.py
```

**FC/DFC Comparison:**
```bash
python scripts/plot_fc.py
python scripts/plot_dfc.py
```

## Visualization Results

### Basic Visualization (`plot.py`)
- Histograms of window and step sizes
- Distribution of MDD and NC data (KDE and scatter plots)
- Epoch-wise changes in learning metrics (accuracy, F1 score, AUC, etc.)

### Ablation Visualization (`plot_ablation.py`)
- Epoch-wise performance comparison (accuracy, F1 score, AUC, reward)
- Performance comparison by ablation mode (bar charts)

## Classifier Architectures

This project supports three classifier architectures:

### 1. CRNN (Convolutional Recurrent Neural Network)
- **Architecture**: Conv1D + GRU (Unidirectional) + FC Layers
- **Features**: Efficient temporal feature extraction
- **Advantages**: Fast training speed, low memory usage
- **Usage**: `--classifier_type crnn`

### 2. CBGRU (Convolutional Bidirectional GRU)
- **Architecture**: Conv1D + Bidirectional GRU + FC Layers
- **Features**: Bidirectional context learning (past↔future)
- **Advantages**: Richer feature representation than CRNN
- **Usage**: `--classifier_type cbgru`

### 3. Spatio-Temporal Transformer
- **Architecture**: Self-Attention Mechanism + Positional Encoding + Transformer Encoder
- **Features**: 
  - Spatial relationship learning (ROI interactions)
  - Temporal pattern learning (long-term dependencies)
  - CLS token-based classification
- **Advantages**: 
  - Full sequence context understanding
  - Interpretability through attention maps
  - Parallel processing capability
- **Usage**: `--classifier_type transformer`

### Comparison

| Feature | CRNN | CBGRU | Transformer |
|---------|------|-------|-------------|
| **Computation** | Low | Medium | High |
| **Memory** | Low | Medium | High |
| **Long-term Dependencies** | Limited | Limited | Excellent |
| **Interpretability** | Low | Low | High |
| **Training Speed** | Fast | Fast | Slow |

## Data Structure

### Dataset Source
**REST-meta-MDD Dataset**: 
- Downloaded from https://rfmri.org/REST-meta-MDD
- Multi-site resting-state fMRI dataset for Major Depressive Disorder research
- Contains both MDD patients and healthy controls (NC)

### Input Data
- **REST-meta-MDD Dataset**: Download from https://rfmri.org/REST-meta-MDD
- Preprocessed fMRI time series and functional connectivity matrices
- Experimental data split for 5-fold cross-validation
- Each fold contains training, validation, and test sets

### Output Data (Generated during execution)
- `results/training_val_test_results_allfolds.csv`: Epoch-wise training, validation, test performance
- `ablation_results/ablation_comparison.csv`: Performance summary by ablation mode and fold
- `ablation_results/ablation_stats.csv`: Mean and standard deviation by ablation mode
- `outputs/*.pkl`: Result files generated during training (reward, window/step usage, etc.)
- `images/*.png`: Generated visualization images (heatmaps, distribution plots, etc.)
- `models/*.pth`: Saved model checkpoints
- `logs/*.log`: Training and execution logs

## ⚙️ Key Hyperparameters

### Training Related
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100000 | Maximum training epochs |
| `--patience` | 10000 | Early stopping patience |
| `--lr` | 1e-5 | Learning rate (common for classifier, macro, micro) |
| `--wd` | 1e-5 | Weight decay |
| `--batch_size` | 1 | Batch size (1 recommended for HRL) |

### RL Related
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--discount_factor` | 0.95 | Gamma (discount factor) |
| `--epsilon` | 0.15 | Epsilon-greedy exploration rate |
| `--rl_batch_size` | 1 | RL training batch size |
| `--buffer_capacity` | 100000 | Replay buffer capacity |
| `--rl_target_update_freq` | 1 | Target network update frequency |

### Model Architecture
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--crnn_hidden_dim` | 128 | Classifier hidden dimension |
| `--crnn_num_layers` | 2 | Classifier number of layers |
| `--crnn_dropout` | 0.2 | Classifier dropout rate |
| `--rl_hidden_dim` | 128 | RL network hidden dimension |
| `--rl_embed_dim` | 128 | RL embedding dimension |
| `--rl_dropout` | 0.2 | RL dropout rate |

### Focal Loss
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--focal_loss_alpha` | 0.75 | Class balancing weight |
| `--focal_loss_gamma` | 3.0 | Focusing parameter |

### Ablation Study
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ablation_mode` | full | Mode: full, no_macro, no_micro, fixed |
| `--fixed_window_size` | 30,60 | Fixed window size (TRs) |
| `--fixed_shift_ratio` | 0.5,1.0 | Fixed shift ratio (0~1) |

## 🔬 HRL Architecture Details

### Hierarchical Structure

```
┌─────────────────────────────────────────────────────┐
│                  HRL System                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐        ┌──────────────┐           │
│  │ Macro-Agent  │        │ Micro-Agent  │           │
│  └──────┬───────┘        └──────┬───────┘           │
│         │                       │                   │
│         ├── Window Size ────────┤                   │
│         │   [10~230 TRs]        │                   │
│         │                       │                   │
│         │                  Shift Ratio              │
│         │                  [0.01~1.00]              │
│         │                       │                   │
│         └───────────┬───────────┘                   │
│                     │                               │
│              ┌──────▼───────┐                       │
│              │  Classifier  │                       │
│              │ CRNN/CBGRU/  │                       │
│              │ Transformer  │                       │
│              └──────────────┘                       │
└─────────────────────────────────────────────────────┘
```

### Reward Function

The reward function is designed to directly optimize for diagnostic utility, with an asymmetric design to penalize misclassifications differently.

**Base Reward (Asymmetric Classification):**
- **True Negative (NC correctly identified):** +1.0
- **True Positive (MDD correctly identified):** +0.5
- **False Positive (NC misclassified as MDD):** -1.0 (Heavier penalty)
- **False Negative (MDD misclassified as NC):** -0.5

**Design Rationale:**
- **Heavier penalty for FP** (-1.0): Avoids unnecessary medical interventions for healthy individuals
- **Lighter penalty for FN** (-0.5): Still clinically significant but less critical than false alarms
- **Lower reward for TP** (+0.5 vs +1.0): Balances the trade-off between sensitivity and specificity

**Confidence Scaling:**
The base reward is scaled by the model's prediction confidence to encourage high-certainty correct predictions.

```python
final_reward = base_reward * (0.75 + 0.5 * confidence)
```

**Scaling Effects:**
- **0% confidence:** Multiplier = 0.75 (minimum scaling)
- **50% confidence:** Multiplier = 1.0 (neutral scaling)
- **100% confidence:** Multiplier = 1.25 (maximum scaling)
- Encourages confident predictions while maintaining stability

## 📊 Experimental Result Format

### Training Results
- `results/training_val_test_results_allfolds.csv`: Epoch-wise performance
- `logs/train_*.log`: Training logs

### Ablation Results
- `ablation_results/ablation_comparison.csv`: Mode-wise comparison
- `ablation_results/ablation_stats.csv`: Mean and standard deviation

### Visualizations
- `figures/*.png`: Learning curves, distributions, heatmaps, etc.

## 🛠️ Requirements

```bash
# Core Requirements (Python 3.8+, recommended 3.12+)
pip install -r requirements.txt

# Key Libraries:
# - PyTorch 2.8.0+ (RTX 5090 optimized)
# - TorchVision 0.23.0+
# - Torch-Geometric 2.6.0+ (Graph Neural Networks)
# - NumPy 1.26.0+, Pandas 2.2.0+, SciPy 1.14.0+
# - Scikit-learn 1.5.0+
# - Matplotlib 3.9.0+, Seaborn 0.13.0+
# - Advanced optimizers and tools
```

**GPU Support:**
- CUDA 12.8+ recommended for RTX 5090
- PyTorch with CUDA support automatically included

## 📝 Citation

**Dataset:**
```
Yan, C. G., Wang, X., Zuo, X. N., & Zang, Y. F. (2016). DPABI: Data Processing & Analysis for (Resting-State) Brain Imaging. 
Neuroinformatics, 14(3), 339-351. https://doi.org/10.1007/s12021-016-9312-x

REST-meta-MDD Project: https://rfmri.org/REST-meta-MDD
```

**Code:**
Add citation information when the paper is published.

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

**Summary:** 
- ✅ Free to use, modify, and share for research/academic purposes
- ❌ Commercial use is not permitted
- 🔗 Must give appropriate credit to the original authors
- 🔄 Derivative works must be licensed under the same terms

## 👥 Contributors

**Chang Hoon Ji**  
Korea University  
Department of Artifical Intellgence  
Email: ckdgns0611@korea.ac.kr

## 📮 Contact

Chang-Hoon Ji: ckdgns0611@korea.ac.kr
