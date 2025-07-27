# FAF5.7 Supply Chain Resilience Analysis Environment Setup

This repository contains a comprehensive machine learning analysis framework for the Freight Analysis Framework (FAF5.7) data to assess supply chain resilience.

## 📋 Prerequisites

- Python 3.8 or higher
- At least 8GB RAM (recommended 16GB for large datasets)
- 5GB free disk space

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

**Windows:**
```cmd
# Double-click or run from command prompt
setup_environment.bat
```

**Linux/macOS:**
```bash
# Make script executable and run
chmod +x setup_environment.sh
./setup_environment.sh
```

### Option 2: Manual Setup

**Step 1: Create Virtual Environment**
```bash
# Windows
python -m venv faf5_env
faf5_env\Scripts\activate

# Linux/macOS
python3 -m venv faf5_env
source faf5_env/bin/activate
```

**Step 2: Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 3: Conda Environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate faf5-supply-chain
```

## 📊 Data Setup

1. Place your `FAF5.7_State.csv` file in the same directory as the notebook
2. Optionally, place `FAF5_metadata.xlsx` for additional reference

## 🔬 Running the Analysis

1. **Activate the environment:**
   ```bash
   # Windows
   faf5_env\Scripts\activate
   
   # Linux/macOS
   source faf5_env/bin/activate
   ```

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open the analysis notebook:**
   - Navigate to `FAF5_Supply_Chain_Resilience_Analysis.ipynb`
   - Run cells sequentially (Cell → Run All)

## 🧠 Machine Learning Models Included

### 1. **Resilience Score Prediction** (Regression)
- **Algorithms:** Random Forest, XGBoost, Gradient Boosting, Linear Regression, Ridge, Neural Network
- **Purpose:** Predict future resilience scores for freight corridors
- **Output:** Continuous resilience scores (0-100)

### 2. **Risk Classification** 
- **Algorithms:** Random Forest, XGBoost, Logistic Regression, Decision Tree, Neural Network
- **Purpose:** Categorize corridors into risk levels
- **Output:** High Risk, Medium-High Risk, Medium-Low Risk, Low Risk

### 3. **Freight Volume Forecasting** (Time Series)
- **Algorithms:** Random Forest, XGBoost, Gradient Boosting, Linear Regression, Neural Network
- **Purpose:** Predict future freight volumes
- **Features:** Trend analysis, volatility, acceleration, seasonal patterns

### 4. **Corridor Clustering** (Unsupervised)
- **Algorithms:** K-Means, DBSCAN, Agglomerative Clustering
- **Purpose:** Identify distinct risk archetypes
- **Output:** Strategic corridor segments

## 📈 Key Features

- **17 Advanced Features:** Including volatility measures, growth trends, concentration risks
- **Cross-Validation:** Robust model validation with 5-fold CV
- **Comprehensive Metrics:** R², RMSE, MAPE, Accuracy, Silhouette Score
- **Interactive Visualizations:** 6-panel performance dashboard
- **Business Impact Analysis:** ROI calculations and deployment recommendations

## 🛠️ Troubleshooting

### Common Issues:

**1. Import Errors:**
```bash
# Reinstall packages
pip install --force-reinstall -r requirements.txt
```

**2. Memory Issues with Large Dataset:**
```python
# Add this to notebook if needed
import pandas as pd
pd.set_option('display.max_columns', None)
# Process data in chunks for very large files
```

**3. XGBoost Installation Issues:**
```bash
# Windows (if Visual Studio issues)
pip install xgboost --no-build-isolation

# macOS (if compiler issues)
brew install libomp
pip install xgboost
```

## 📁 File Structure

```
FAF5.7_State/
├── FAF5.7_State.csv                           # Main dataset
├── FAF5_metadata.xlsx                         # Metadata (optional)
├── FAF5_Supply_Chain_Resilience_Analysis.ipynb # Main analysis notebook
├── requirements.txt                           # Python dependencies
├── environment.yml                           # Conda environment
├── setup_environment.bat                     # Windows setup script
├── setup_environment.sh                      # Linux/macOS setup script
└── README.md                                 # This file
```

## 💡 Usage Tips

1. **Start Small:** Test with a sample of data first if working with the full 503MB dataset
2. **Monitor Memory:** Use Task Manager/Activity Monitor to check memory usage
3. **Save Progress:** Use Jupyter's checkpoint feature frequently
4. **Export Results:** Models and results can be saved using pickle for later use

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.9+ |
| RAM | 8GB | 16GB+ |
| Storage | 5GB | 10GB+ |
| CPU | 4 cores | 8+ cores |

## 🆘 Support

If you encounter issues:

1. Check that all dependencies are installed correctly
2. Verify your data files are in the correct location
3. Ensure you have sufficient RAM for the dataset size
4. Try running cells individually rather than "Run All"

## 🎯 Expected Results

The analysis will provide:
- Resilience scores for all freight corridors
- Risk classifications and recommendations
- Volume forecasts for capacity planning
- Strategic clustering insights for diversification
- Comprehensive deployment roadmap with ROI projections

## ⚡ Performance Notes

- Full analysis on 1.2M records takes approximately 15-30 minutes
- XGBoost models may take longer but provide best performance
- Consider using smaller sample for initial testing
- GPU acceleration available for XGBoost if CUDA is installed 