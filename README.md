# Disaster Tweet Classification

This repository is **optional** material for our final report, which contains code for classifying disaster-related tweets using various machine learning approaches including Logistic Regression, Random Forest, and BERT-based models.

## Dataset

The dataset and results can be found at:
https://drive.google.com/drive/folders/112ZGffGP_tbNO2qO2OWy2elOGOXqxF0w?usp=sharing

## Project Structure

```
├── assets/
│   ├── Figure_1.png     # Document length statistics
│   ├── Figure_2.png     # Target label distribution
│   ├── Figure_3.png     # Common terms visualization
│   ├── Figure_4.png     # Word importance visualization
│   └── Figure_5.png     # Additional visualization
├── code/
│   ├── bert_model.py    # BERT-based classification implementation
│   ├── lg_rf.py         # Logistic Regression and Random Forest implementation
│   └── eda.py           # Exploratory Data Analysis code
└── README.md            # This file
```

## Models Implemented

1. **Logistic Regression**
   - TF-IDF based feature extraction (5000 dimensions)
   - Basic text preprocessing
   - Implementation using scikit-learn

2. **Random Forest**
   - Same feature extraction as Logistic Regression
   - 200 trees in the forest
   - Implementation using scikit-learn

3. **BERT-based Classification**
   - Feature extraction using BERT
   - AdamW optimizer
   - Batch size: 32
   - Learning rate: 1e-5
   - Implementation using HuggingFace transformers

## Results

| Model name | Precision | Recall | F1 |
|------------|-----------|---------|-----|
| Logistic Regression | 0.81 | 0.787 | 0.797 |
| Random Forest | 0.797 | 0.77 | 0.773 |
| BERT-based Classification | 0.833 | 0.827 | 0.83 |

## References

All code implementations are based on the project report and follow the methodologies described therein. For detailed methodology and analysis, please refer to the project [report](https://drive.google.com/file/d/1Bs2CcmJj7or29SazYw5FpnHpWQaUrBwP/view?usp=sharing).


## Usage

1. Download the dataset from the provided Google Drive link
2. Run EDA:
   ```
   python code/eda.py
   ```
3. Train and evaluate models:
   ```
   python code/lg_rf.py      # For Logistic Regression and Random Forest
   python code/bert_model.py # For BERT-based classification
   ```
