# PreMortemML – Data Insight Platform

PreMortemML is a data quality analysis platform designed to evaluate machine learning datasets before model training begins. The goal is to identify potential issues in the dataset that could negatively affect model performance, reliability, and downstream decision-making.

The platform performs automated checks for common data problems such as outliers, duplicate records, and class imbalance, allowing practitioners to inspect dataset quality early in the ML workflow.

Built as an interactive web interface using Streamlit, PreMortemML enables users to upload datasets, configure feature selections, run automated analysis, and review detected issues through structured reports and visual outputs.

---

# Overview

Machine learning performance is heavily dependent on dataset quality. Problems such as noisy samples, duplicated data points, or abnormal values can significantly degrade model performance.

PreMortemML provides a structured workflow for examining datasets before training begins. By detecting and summarizing potential issues, it helps practitioners understand the dataset and decide whether additional cleaning or preprocessing is required.

---

# Key Features

## 1. Dataset Upload and Configuration

Users can upload CSV datasets directly through the web interface. The system automatically reads the dataset structure and allows users to configure the analysis parameters.

Capabilities include:

* Upload CSV datasets through the interface
* Automatic detection of numeric feature columns
* Manual selection of label and feature columns
* Dataset preview before running analysis

### Interface

```markdown
![Dataset Upload](assets/Screenshot_2026-03-16_at_5.21.40_PM-7c60dd03-5292-4995-a235-bc76f1ea8223.png)
```

---

## 2. Automated Data Quality Analysis

Once configured, the platform runs automated checks to detect common data quality issues.

The analysis currently includes detection of:

* Outliers in numeric feature columns
* Near-duplicate samples
* Class imbalance in the label column
* Dataset structural statistics

The analysis summarizes the detected issues and provides an overview of dataset health.

### Example Results Dashboard

```markdown
![Quality Analysis Results](assets/Screenshot_2026-03-16_at_5.23.26_PM-c03c0faf-d6f6-4298-a673-9f7edb2af3d5.png)
```

---

## 3. Issue Inspection and Detailed Reports

After running the analysis, users can inspect detected issues in detail.

The platform provides:

* Summary statistics of detected problems
* Issue counts for each category
* Row-level inspection of problematic samples
* Exportable issue reports

Users can also download the detected issues as a CSV file for further investigation.

### Detailed Issue View

```markdown
![Issue Details](assets/Screenshot_2026-03-16_at_5.23.05_PM-0a08cb77-22ba-4b6a-a6e6-e6d01f7ea452.png)
```

You can additionally show the CSV export preview:

```markdown
![Quality Analysis CSV](assets/Screenshot_2026-03-16_at_5.23.01_PM-2b274b84-2bd8-4b03-882f-5302624299b4.png)
```

---

## 4. Configurable Analysis Settings

The platform allows users to adjust analysis behavior through configuration options.

These settings include:

* Cross-validation fold configuration
* Sampling options for large datasets
* Missing value handling
* Feature scaling
* Selection of issue detection modules

### Settings Panel

```markdown
![Settings Panel](assets/Screenshot_2026-03-16_at_5.22.10_PM-461b0a11-148d-420e-961f-21227d5cab49.png)
```

You can also include an overall landing view:

```markdown
![Landing View](assets/Screenshot_2026-03-16_at_5.18.16_PM-35579390-810d-4f4d-9c8c-e0854e285b99.png)
```

---

# Tech Stack

## Core Platform

* Python 3.10+
* Streamlit – interactive web interface

## Data Processing and Analysis

* pandas – dataset handling and manipulation
* numpy – numerical operations
* scikit-learn – preprocessing utilities and validation tools

## Visualization

* matplotlib
* seaborn

## Utility Libraries

* tqdm – progress tracking for long operations
* termcolor – formatted terminal output

---

# Project Structure

```text
.
├── data_insight_platform.py
├── premortemml/
│   └── core data analysis package
├── platform_requirements.txt
├── start_platform.sh
└── README.md
```

---

# Installation

Install the required dependencies:

```bash
pip install -r platform_requirements.txt
pip install -e .
```

---

# Running the Platform

Start the Streamlit application:

```bash
streamlit run data_insight_platform.py
```

Or use the provided script:

```bash
./start_platform.sh
```

Once the server starts, open:

```text
http://localhost:8501
```

Upload a dataset and run the analysis through the interface.

---

# Author

Built by **amukta14**
