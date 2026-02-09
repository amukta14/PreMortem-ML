# PreMortemML - Data Insight Platform

**Before you train anything, understand what could go wrong.**

A comprehensive pre-training risk analysis and data quality assessment platform that helps you identify potential issues in your ML datasets before model training begins.

## Vision

PreMortemML provides a "pre-mortem" analysis for machine learning projects—identifying potential failures and risks before they happen. The platform combines automated data quality detection with a unique **Cost-of-Mistakes Simulator** that estimates expected errors and their costs based on dataset characteristics, helping you make informed decisions about whether and how to proceed with model training.

## Key Features

- **Automated Data Quality Analysis**: Detect label errors, outliers, duplicates, class imbalance, and other data issues
- **Cost-of-Mistakes Simulator**: Estimate expected false positives/negatives and their costs before training
- **Confusion-Risk Matrix**: Visualize expected model performance and error patterns
- **Cost Sensitivity Analysis**: Understand how different error costs impact your decision-making
- **Dynamic Dataset Handling**: Efficiently processes datasets of any size with intelligent sampling and cross-validation

## Tech Stack

### Core Platform
- **Python 3.10+**: Primary programming language
- **Streamlit**: Web application framework for interactive dashboards
- **premortemml**: Custom Python package for data quality analysis and issue detection

### Data Processing & ML
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models (LogisticRegression, RandomForestClassifier) and utilities (cross-validation, preprocessing)
- **matplotlib** & **seaborn**: Data visualization

### Additional Dependencies
- **datasets**: Dataset handling utilities
- **tqdm**: Progress bars for long-running operations
- **termcolor**: Terminal output formatting

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r platform_requirements.txt
   pip install -e .  # Install premortemml package
   ```

2. **Run the platform**:
   ```bash
   ./start_platform.sh
   ```
   Or directly:
   ```bash
   streamlit run data_insight_platform.py
   ```

3. **Access the web interface**:
   - Open your browser to `http://localhost:8501`
   - Upload your dataset (CSV format)
   - Select features and label column
   - Run quality analysis or cost simulation

## Project Structure

```
.
├── data_insight_platform.py    # Main Streamlit application
├── premortemml/                 # Core analysis package
├── platform_requirements.txt   # Web app dependencies
├── start_platform.sh            # Launch script
└── README.md                   # This file
```

## Usage

### Quality Analysis
Upload your dataset and let the platform automatically detect:
- Label errors and inconsistencies
- Outliers and anomalous examples
- Near-duplicate entries
- Class imbalance issues
- Missing values and data quality problems

### Cost Simulator
Before training, simulate:
- Expected confusion matrix
- False positive/negative rates
- Cost sensitivity curves
- "Should You Even Model This?" risk score
- Plain-English warnings about potential issues

## Built By

**amukta14** - [GitHub](https://github.com/amukta14)

---

*PreMortemML: Because prevention is better than debugging.*
