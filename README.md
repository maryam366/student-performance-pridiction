# student-performance-pridiction
## 📌 Project Overview

This project builds a complete Machine Learning pipeline to **predict whether a student will Pass or Fail** their final exam based on demographic, social, and academic features.

Using the **UCI Student Performance Dataset** — real data collected from 1,044 students across Math and Portuguese courses in two Portuguese secondary schools — three classification models were trained, tuned, and evaluated.

**🏆 Best Model: Decision Tree — 93.3% Accuracy | F1 = 0.957 | ROC-AUC = 0.942**

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── data/
│   ├── student-mat.csv         # Math course (395 students)
│   └── student-por.csv         # Portuguese course (649 students)
│
├── src/
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── train_evaluate.py       # Model training, tuning & evaluation
│   └── predict.py              # Single student inference
│
├── outputs/
│   ├── eda_overview.png
│   ├── feature_insights.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
├── models/
│   └── best_model_Decision_Tree.pkl
│
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

**Source:** [UCI ML Repository — Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

**Citation:** P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* EUROSIS, 2008.

| Property | Value |
|---|---|
| Students | 1,044 (Math: 395 + Portuguese: 649) |
| Features | 33 (demographic, social, academic) |
| Target | Pass (G3 ≥ 10) / Fail (G3 < 10) |
| Missing values | None |
| Pass rate | Math: 67.1% / Portuguese: 84.6% |

### Key Features

| Feature | Description |
|---|---|
| `studytime` | Weekly study time (1=<2h → 4=>10h) |
| `failures` | Number of past class failures |
| `higher` | Wants to pursue higher education |
| `Medu / Fedu` | Mother's / Father's education level |
| `absences` | School absences |
| `Dalc / Walc` | Weekday / Weekend alcohol consumption |
| `G1, G2` | 1st and 2nd term grades |
| `G3` | Final grade → used to create Pass/Fail target |

---

## 📊 EDA Highlights

Key findings from real data:

- **G1 and G2** are the strongest predictors (correlation with G3 = 0.809)
- Students with **0 past failures** pass at 85% rate vs 44.8% for those with failures
- Students who **want higher education** pass at 80.7% rate
- **Portuguese** students pass at a higher rate (84.6%) than Math students (67.1%)
- **Alcohol consumption** (especially weekday) negatively impacts final grades

![EDA Overview](outputs/eda_overview.png)
![Feature Insights](outputs/feature_insights.png)

---

## 🤖 Models & Results

Three classifiers trained using Scikit-learn Pipelines with **StandardScaler** preprocessing and **GridSearchCV (5-fold CV)** for hyperparameter tuning:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Decision Tree** ⭐ | **93.3%** | **96.3%** | **95.1%** | **0.957** | 0.942 |
| Logistic Regression | 91.9% | 95.6% | 93.9% | 0.947 | **0.962** |
| KNN | 80.9% | 82.2% | 96.3% | 0.887 | 0.859 |

### Best Hyperparameters (GridSearchCV)

| Model | Best Parameters |
|---|---|
| Decision Tree | `max_depth=3, criterion='gini', min_samples_split=2` |
| Logistic Regression | `C=100, penalty='l1', solver='liblinear'` |
| KNN | `n_neighbors=9, weights='distance', metric='euclidean'` |

![Model Comparison](outputs/model_comparison.png)
![Confusion Matrices](outputs/confusion_matrices.png)
![ROC Curves](outputs/roc_curves.png)
![Feature Importance](outputs/feature_importance.png)

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/maryam366/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run EDA
```bash
python src/eda.py
```

### 4. Train & evaluate models
```bash
python src/train_evaluate.py
```

### 5. Predict for a new student
```bash
python src/predict.py
```
### 6.Run app
https://student-performance-pridiction-d66wtbjm9tchwepfa6sp69.streamlit.app/
---

## 💡 Key Learnings

- **Real data is messy**: two separate CSV files with different delimiters (`;`) needed careful loading and merging
- **G1 and G2 dominate**: early term grades are extremely predictive of final outcomes — this is expected but confirms the model is learning correctly
- **Hyperparameter tuning mattered most for Decision Tree**: accuracy improved from 90.4% → 93.3% after tuning
- **Class imbalance (78% Pass)**: monitored using F1-score and per-class precision/recall to avoid misleading accuracy numbers
- **Pipeline design**: using Scikit-learn Pipelines prevents data leakage by applying scaling only on training data during CV

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Scikit-learn** — Pipelines, GridSearchCV, DecisionTree, KNN, LogisticRegression
- **Pandas & NumPy** — data loading, manipulation
- **Matplotlib & Seaborn** — EDA and evaluation visualizations
- **Joblib** — model serialization

---

## 📜 License

MIT License. Dataset is publicly available via UCI ML Repository.

---

*Built as a 6th semester CS Machine Learning portfolio project using real academic data.*

