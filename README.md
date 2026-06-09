# Credit Card Fraud Detection
## End-to-End Machine Learning Pipeline

**Author:** Saad Ahsan  
**GitHub:** [github.com/saadahsan0009](https://github.com/saadahsan0009)  
**LinkedIn:** [linkedin.com/in/saad-ahsan-1901sam](https://www.linkedin.com/in/saad-ahsan-1901sam/)

---

## 📌 The Problem

Credit card fraud costs the global economy billions every year. The challenge isn't just building a model that detects fraud — it's building one that works on data where fraud represents only **0.172% of all transactions**. A model that predicts every transaction as legitimate would be 99.83% accurate and completely useless.

This project tackles that problem properly — handling the class imbalance, comparing multiple approaches, and making the final model's decisions explainable.

---

## 🎯 Results

| Model | ROC-AUC | Recall | F1 Score | Avg Precision |
|-------|---------|--------|----------|---------------|
| Logistic Regression | ~0.97 | — | — | — |
| Random Forest | ~0.98 | — | — | — |
| **XGBoost (best)** | **~0.98** | **High** | **Strong** | **Best** |

> Evaluation focused on **Recall** and **ROC-AUC** rather than accuracy — missing a fraud is far more costly than a false alarm.

---

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions over two days (September 2013)
- **Fraud rate:** 492 frauds out of 284,807 transactions — **0.172%**
- **Features:** 30 features — Time, Amount, and V1–V28 (PCA-transformed for privacy)
- **Target:** Class (0 = legitimate, 1 = fraud)

> The dataset is not included in this repository due to size. Download it directly from Kaggle using the link above.

---

## ⚙️ What This Project Does

### The class imbalance problem
With 0.172% fraud rate, standard ML models simply learn to predict everything as legitimate. We address this with **SMOTE (Synthetic Minority Oversampling Technique)** — which creates synthetic fraud samples by interpolating between existing ones, giving the model enough fraud examples to learn meaningful patterns.

SMOTE is applied **only on training data** — never on the test set, which must reflect real-world conditions.

### Three models compared
Starting simple and increasing complexity:

1. **Logistic Regression** — fast, interpretable baseline
2. **Random Forest** — ensemble of trees, handles non-linearity well
3. **XGBoost** — gradient boosting, sequentially corrects errors, best on tabular data

### Evaluation approach
For imbalanced data, accuracy is misleading. We use:
- **ROC-AUC** — overall discrimination across all thresholds
- **Precision-Recall curves** — more informative than ROC for imbalanced problems
- **Recall** — what fraction of actual frauds did we catch?
- **F1-Score** — balance between precision and recall
- **Confusion matrices** — exactly what the model gets right and wrong

### SHAP explainability
A fraud flag without a reason is hard to act on. **SHAP (SHapley Additive exPlanations)** shows which features pushed each prediction toward fraud and by how much — making every alert defensible and auditable.

---

## 📁 Repository Structure

```
├── fraud_detection.ipynb    # Full analysis notebook
├── figures/                 # All generated plots
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── time_patterns.png
│   ├── feature_distributions.png
│   ├── roc_pr_curves.png
│   ├── confusion_matrices.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_waterfall.png
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange)
![SHAP](https://img.shields.io/badge/SHAP-explainability-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-red)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-purple)

- **Language:** Python 3.11
- **ML:** XGBoost, Scikit-learn, Random Forest, Logistic Regression
- **Imbalance handling:** imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Data:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/saadahsan0009/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# Create a clean environment
conda create -n fraud_env python=3.11
conda activate fraud_env

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn==1.3.2 xgboost==1.7.6 imbalanced-learn shap jupyter

# Download the dataset from Kaggle and place creditcard.csv in this folder
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Launch notebook
jupyter notebook fraud_detection.ipynb
```

---

## 💡 Key Takeaways

**1. Accuracy is the wrong metric here.** A model predicting everything as legitimate gets 99.83% accuracy. We need recall, precision, and AUC.

**2. SMOTE works.** Balancing the training set from 577:1 to 1:1 gave all three models meaningful signal to learn from.

**3. XGBoost wins on tabular data.** Its sequential error-correction approach captures non-linear patterns in the PCA-transformed features that logistic regression cannot.

**4. Explainability matters in production.** SHAP makes alerts auditable — which features drove the flag, and by how much. This is critical for compliance and investigator workflows.

**5. Threshold matters.** In production, the classification threshold would be tuned based on the business cost of missing fraud versus the cost of false alarms — not just optimised for F1.

---

## ⚠️ Production Considerations

- **Threshold tuning:** Lower threshold = higher recall, more false alarms. Tune based on fraud cost vs. operational cost.
- **Model monitoring:** Fraud patterns evolve. Models need regular retraining on fresh data.
- **Feature drift:** Transaction patterns change over time — monitor distributions continuously.
- **Real-time scoring:** XGBoost inference is fast enough for real-time transaction screening.

---

## 👤 Author

**Saad Ahsan**  
MSc Data Science (Distinction) — Middlesex University London  
MSc Electronic Engineering (Distinction) — University of Essex  

📧 saadahsan0009@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/saad-ahsan-1901sam/)  
🐙 [GitHub](https://github.com/saadahsan0009)
