# Semantic-Aware Reinforcement Learning for Resource-Efficient IoT Communication

## 📋 Project Overview

This repository implements a comprehensive framework for optimizing IoT communication using semantic-aware reinforcement learning. The system combines Deep Q-Networks (DQN), machine learning classifiers, and anomaly detection to improve resource efficiency in Sigfox-based IoT networks in Antwerp.

## 🏗️ System Architecture

The project follows a modular pipeline architecture:

```
Data Preprocessing → Model Training → Evaluation → Visualization → Explainable AI Analysis
```

## 📁 Project Structure

```
├── main.py                          # Main pipeline execution script
├── preprocessing.py                 # Data preprocessing and feature engineering
├── models.py                        # Neural network models and environment
├── training.py                      # Model training routines
├── evaluation.py                    # Performance evaluation metrics
├── visualization.py                 # Plotting and visualization utilities
├── utils.py                         # Utility functions and helpers
├── Action_and_reward_distribution.py # Reward and action analysis
├── shap_analysis.py                 # SHAP-based model interpretability
├── xai_analysis.py                  # Comprehensive XAI analysis
└── data/
    ├── sigfox_dataset_antwerp.csv   # Raw Sigfox dataset
    ├── sigfox_bs_mapping.csv        # Base station mapping
    └── semantic_features_antwerp.csv # Processed semantic features
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn shap lime
```

### Basic Usage

1. **Run the complete pipeline:**
```bash
python main.py
```

2. **Perform reward and action analysis:**
```bash
python Action_and_reward_distribution.py
```

3. **Generate SHAP explanations:**
```bash
python shap_analysis.py
```

4. **Run comprehensive XAI analysis:**
```bash
python xai_analysis.py
```

## 🔧 Core Components

### 1. Data Preprocessing (`preprocessing.py`)
- Cleans and processes raw Sigfox data
- Extracts semantic features (RSSI, active base stations, location, time)
- Normalizes features for model training
- Generates classification labels based on connection quality

### 2. Machine Learning Models (`models.py`)
- **DQN Network**: Deep Q-Network for reinforcement learning
- **Random Forest**: Classification model for connection quality prediction
- **Isolation Forest**: Anomaly detection for network monitoring
- **Environment Simulator**: Custom RL environment for IoT communication

### 3. Training Framework (`training.py`)
- DQN training with experience replay and target networks
- Supervised learning for classification tasks
- Anomaly detection model training
- Hyperparameter configuration

### 4. Evaluation Suite (`evaluation.py`)
- DQN agent performance evaluation
- Classification metrics (accuracy, ROC-AUC, confusion matrix)
- Anomaly detection performance assessment
- Statistical analysis of model outputs

### 5. Visualization Tools (`visualization.py`)
- Training progress monitoring
- Model performance dashboards
- Geographic anomaly mapping
- Feature importance plots
- ROC curve analysis

## 📊 Key Features

### Reinforcement Learning
- **State Space**: Mean RSSI, active base stations, location, time
- **Action Space**: [Do Nothing, Move North, Move South, Move East, Move West]
- **Reward Function**: Connection quality + time efficiency - movement cost

### Machine Learning
- **Binary Classification**: Good vs Poor connection quality
- **Feature Set**: 5 semantic features for comprehensive analysis
- **Model Validation**: Cross-validation and test set evaluation

### Explainable AI
- **SHAP Analysis**: Model interpretability and feature contributions
- **LIME Explanations**: Local interpretable model explanations
- **Partial Dependence**: Feature relationship analysis
- **Permutation Importance**: Feature significance assessment

## 📈 Outputs and Visualizations

The system generates publication-ready visualizations:

1. **Reward Distribution Analysis**
   - Overall reward distribution
   - Optimal reward analysis
   - Action-specific reward patterns
   - Cumulative reward probabilities

2. **Action Distribution Analysis**
   - Average reward by action type
   - Action selection by connection quality
   - Geographic action patterns

3. **Model Performance Metrics**
   - Training progress curves
   - Confusion matrices
   - ROC curves and AUC scores
   - Feature importance rankings

4. **Explainable AI Reports**
   - SHAP summary plots
   - Feature dependence analysis
   - Local instance explanations
   - Model decision transparency

## 🎯 Use Cases

### Network Optimization
- Dynamic resource allocation based on semantic context
- Predictive maintenance through anomaly detection
- Energy-efficient device positioning strategies

### Quality of Service
- Real-time connection quality prediction
- Proactive network management
- Geographic coverage optimization

### Research and Analysis
- Semantic feature importance analysis
- Reinforcement learning policy evaluation
- Network performance benchmarking

## ⚙️ Configuration

### Key Parameters

**DQN Training:**
```python
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
```

**Feature Engineering:**
- Mean RSSI threshold: -110 dBm
- Active base stations threshold: 3
- Time features: Hour of day
- Location features: Latitude, Longitude

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{semantic_iot_rl,
  title = {Semantic-Aware Reinforcement Learning for Resource-Efficient IoT Communication},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo/semantic-iot-rl}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in each module
- Review the example outputs in the `results/` directory

## 🔄 Updates

- **v1.0**: Initial release with complete pipeline
- **v1.1**: Added SHAP and XAI capabilities
- **v1.2**: Enhanced visualization and documentation

---

*This project demonstrates advanced techniques in IoT network optimization using semantic-aware machine learning and reinforcement learning approaches.*
