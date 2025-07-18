# ML Logistics Optimization using XGBoost & Reinforcement Learning

⚠️ **PROJECT DISCLAIMER** ⚠️

This repository is based on a real-world logistics optimization project, but:
- **No actual company data or proprietary logic is included**
- **Core optimization frameworks and architectures are abstracted for illustrative purposes**
- **All data, business rules, and parameters have been modified and abstracted for confidentiality**

The project demonstrates advanced machine learning and reinforcement learning methodologies for logistics optimization without using any proprietary business information.

🚛 **Project Overview**

This repository contains a comprehensive solution for logistics optimization using advanced machine learning techniques. The primary goal was to optimize trailer allocation and relocation decisions across a nationwide network while maximizing operational efficiency and minimizing costs. Our solution combines **XGBoost for demand forecasting** and **Reinforcement Learning for dynamic optimization**.

We implemented a multi-phase approach:
- **Phase 1**: Synthetic data augmentation and demand forecasting using XGBoost/LightGBM
- **Phase 2**: Rule-based optimization for initial trailer allocation
- **Phase 3**: Reinforcement Learning for dynamic optimization and real-time decision making

📌 **Motivation & Technology**

**Why XGBoost for Demand Forecasting?**
- **Efficiency**: Handles large time-series datasets quickly with parallel processing
- **Accuracy**: Provides superior predictive performance for demand patterns
- **Robustness**: Handles missing values and categorical features efficiently
- **Feature Importance**: Offers interpretable insights into demand drivers

**Why Reinforcement Learning for Optimization?**
- **Dynamic Decision Making**: Adapts to changing demand patterns and network conditions
- **Multi-Objective Optimization**: Balances revenue maximization, cost minimization, and service levels
- **Real-time Adaptation**: Continuously learns and improves from operational feedback
- **Complex Constraint Handling**: Manages multiple operational constraints simultaneously

**Why Rule-Based + RL Hybrid Approach?**
- **Rule-Based**: Provides interpretable baseline and business logic foundation
- **RL Enhancement**: Adds adaptive learning and optimization capabilities
- **Risk Management**: Combines proven business rules with advanced optimization

📖 **Table of Contents**

- [ML Logistics Optimization using XGBoost \& Reinforcement Learning](#ml-logistics-optimization-using-xgboost--reinforcement-learning)
    - [Step 1: Synthetic Data Augmentation](#step-1-synthetic-data-augmentation)
    - [Step 2: ML Demand Forecasting](#step-2-ml-demand-forecasting)
    - [Step 3: Rule-Based Optimization Pseudocode](#step-3-rule-based-optimization-pseudocode)
    - [Step 4: RL Optimization](#step-4-rl-optimization)
    - [ML Forecasting Results](#ml-forecasting-results)
    - [Rule-Based Optimization Results](#rule-based-optimization-results)
    - [Reinforcement Learning Results](#reinforcement-learning-results)
  - [Installation \& Setup](#installation--setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)

📚 **Repository Contents**

```
ML_Logistics_Optimization/
├── 1_XGBoost_Modeling.py                         # XGBoost demand forecasting
├── 1_LightGBM_Modeling.py                        # LightGBM demand forecasting
├── 2_Rule_Based_Optimization_Pseudocode.md       # Rule-based optimization logic
├── 3_RL_Logistics_Optimization.py                # RL optimization system
├── dataset/                                       # Synthetic datasets
│   ├── Synthetic_Demand_TimeSeries.csv
│   ├── MarketIndex_Comdist_Synthetic.csv
│   ├── NetLoc_Synthetic_cleaned.csv
│   ├── Telematics_Synthetic.csv
│   └── Enhanced_Telematics_Data.csv
├── Output/                                        # Results and models
│   ├── Regional_Demand_Index.csv
│   ├── financial_analysis_results.png
│   ├── rl_vs_rule_based_comparison.png
│   └── rl_training_progress.png
├── Paper/                                        # Documentation
├── Poster_INFORMS/ 
├── requirements_ml.txt                           # ML dependencies
└── requirements_rl.txt                           # RL dependencies
```

🚀 **Project Workflow**

### Step 1: Synthetic Data Augmentation

**Demand Data Generation** (`0_Synthetic_Data_Augmentation(Demand).py`)
- Generated realistic demand time series with seasonal patterns
- Created regional demand variations and market tier classifications
- Implemented trend analysis and forecasting capabilities
- **Output**: `dataset/Synthetic_Demand_TimeSeries.csv`

**Network Data Generation** (`0_Synthetic_Data_Augmentation(Network).py`)
- Created synthetic network locations with geographic distribution
- Generated telematics data for trailer positioning and movement
- Established realistic operational constraints and capacity limits
- **Output**: `dataset/NetLoc_Synthetic_cleaned.csv`, `dataset/Telematics_Synthetic.csv`

### Step 2: ML Demand Forecasting

**XGBoost Implementation** (`1_XGBoost_Modeling.py`)
- **Feature Engineering**: Calendar features, lag variables, regional encoding
- **Hyperparameter Tuning**: Optuna optimization for model performance
- **Model Evaluation**: RMSE, MAE, MAPE, and R² metrics
- **SHAP Analysis**: Feature importance and interpretability
- **Output**: `Output/Regional_Demand_Index.csv` - Regional demand index predictions for each location

**LightGBM Implementation** (`1_LightGBM_Modeling.py`)
- Alternative gradient boosting approach for comparison
- Optimized for speed and memory efficiency
- Cross-validation and ensemble methods

### Step 3: Rule-Based Optimization Pseudocode

**Core Algorithm** (`2_Rule_Based_Optimization_Pseudocode.md`)
- **Network Allocations**: Geographic distribution based on demand tiers
- **Trailer Relocations**: Cost-minimizing relocation strategies
- **Business Rules**: Market tier classification, capacity constraints
- **Financial Modeling**: Revenue optimization and cost minimization

**Key Features:**
- Multi-tier market classification (Tier 1, 2, 3)
- Geographic proximity optimization using Haversine distance
- Capacity constraints and operational limits
- Revenue and cost modeling with utilization rates

**Output**: `Output/financial_analysis_results.png` - Comprehensive financial analysis across different numbers of cities

### Step 4: RL Optimization

**RL Environment** (`3_RL_Logistics_Optimization.py`)
- **State Space**: Trailer positions, demand patterns, network capacity
- **Action Space**: Trailer relocation and allocation decisions
- **Reward Function**: Multi-objective optimization (revenue, cost, service level)
- **DQN Agent**: Deep Q-Network for policy learning

**Advanced Features:**
- Custom Gym environment for logistics optimization
- Deep Q-Network with experience replay
- Multi-objective reward function
- Real-time adaptation to changing conditions

**Outputs**: 
- `Output/rl_training_progress.png` - Training progress visualization
- `Output/rl_vs_rule_based_comparison.png` - Performance comparison between RL and rule-based approaches

📊 **Results & Performance**

### ML Forecasting Results
- **XGBoost Performance**: RMSE < 0.15, R² > 0.85
- **Feature Importance**: Regional factors, seasonal patterns, lag variables
- **Model Interpretability**: SHAP analysis for business insights
- **Output**: Regional demand index predictions for strategic planning

### Rule-Based Optimization Results
- **Trailer Allocation**: Optimized distribution across multiple target cities
- **Cost Reduction**: 15-20% reduction in relocation costs
- **Service Level**: Maintained 95%+ service level across network
- **Multi-Scenario Analysis**: Comprehensive financial analysis for 1-30 cities

### Reinforcement Learning Results
- **Learning Convergence**: Stable policy learning within 500 episodes
- **Performance Improvement**: 25-30% better than rule-based baseline
- **Adaptive Capability**: Real-time optimization under changing conditions
- **Visualization**: Training progress and performance comparison charts

🖼️ **Project Documentation**

For comprehensive project documentation, please see:
- 📃 **Final Paper**: `Paper/Final_Paper_Logistics.pdf`
- 📊 **Conference Poster**: `Poster_INFORMS/Final_Poster_Logistics.pdf`

These documents provide detailed methodology, results analysis, and business impact assessment.

📋 **Usage Guide**

1. **Data Preparation**
   ```bash
   python 0_Synthetic_Data_Augmentation(Demand).py
   python 0_Synthetic_Data_Augmentation(Network).py
   ```

2. **Demand Forecasting**
   ```bash
   python 1_XGBoost_Modeling.py
   # or
   python 1_LightGBM_Modeling.py
   ```

3. **Rule-Based Optimization**
   ```bash
   python 2_Rule-Based_Optimization.py
   ```

4. **Reinforcement Learning Optimization**
   ```bash
   python 3_RL_Logistics_Optimization.py
   ```

5. **View Results**
   - Check `Output/Regional_Demand_Index.csv` for demand predictions
   - Check `Output/financial_analysis_results.png` for financial results of rule-based optimization
   - Check `Output/rl_training_results.png` for RL training process
   - Check `Output/rl_vs_rule_based_comparison.png` for RL comparison results

📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for RL training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/LeannaJ/ML_Logistics_Optimization.git
cd ML_Logistics_Optimization
```

2. **Install ML dependencies**
```bash
pip install -r requirements_ml.txt
```

3. **Install RL dependencies**
```bash
pip install -r requirements_rl.txt
```

### Usage

1. **Run ML Forecasting Pipeline**
```bash
python 1_XGBoost_Modeling.py
```

2. **Run Rule-Based Optimization**
2_Rule-Based_Optimization_Pseudocod.md


3. **Run RL Optimization**
```bash
python 3_RL_Logistics_Optimization.py
```

4. **View Results**
- Check `Output` folder



🤝 **Citation**

If you find this repository helpful in your research, teaching, or other work, please consider citing or linking back to the repository:

```
ML Logistics Optimization Project. (2025). 
ML Logistics Optimization using XGBoost & Reinforcement Learning. 
GitHub Repository: https://github.com/LeannaJ/ML_Logistics_Optimization
```

---

**Created by Leanna Seung-mi Jeon with 💻** 
