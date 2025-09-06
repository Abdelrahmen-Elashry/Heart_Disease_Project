# 🩺 Heart Disease Prediction Project – Full Machine Learning Pipeline  

---

## 📌 Project Description
This project focuses on building a **comprehensive machine learning pipeline** to predict the presence of heart disease based on patient health records. It covers the **entire workflow**: from data preprocessing, dimensionality reduction, feature selection, supervised and unsupervised learning, model evaluation, optimization, to final deployment. The ultimate goal is to help identify patients at risk of heart disease through interpretable and efficient ML models.

---

## 📊 Dataset Information
- **Source**: [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)  
- **Rows**: 303  
- **Columns**: 14 (13 features + 1 target)  
- **Target Variable**: `num`
- **Missing Values**: Present in some columns (`ca`, `thal`)  
- **Not Included Columns**: Only the 13 main features were used.

---

## 🛠️ Tools & Technologies
- **Programming Language**: Python  
- **Libraries**:  
  - Data Handling: Pandas, NumPy  
  - Visualization: Matplotlib, Seaborn, plotly  
  - Machine Learning: Scikit-learn  

---

## 🔄 Workflow Steps
1. **Data Preprocessing & Cleaning** – Handled missing values, encoded categorical variables, scaled numerical features, and performed EDA.  
2. **Dimensionality Reduction (PCA)** – Reduced features while retaining most variance.  
3. **Feature Selection** – Used RFE, Chi-Square, and feature importance to keep relevant predictors.  
4. **Supervised Learning** – Trained Logistic Regression, Decision Tree, Random Forest, and SVM.  
5. **Unsupervised Learning** – Applied K-Means and Hierarchical Clustering to detect patterns.  
6. **Model Evaluation** – Measured Accuracy, Precision, Recall, F1-score, and AUC.  
7. **Hyperparameter Tuning** – Applied GridSearchCV and RandomizedSearchCV for optimization.  
8. **Model Export & Deployment** – Saved final model using pickle.

---

## 📈 Models and Metrics

| Model                | Accuracy | Precision | Recall | F1-Score | AUC  |
|-----------------------|----------|-----------|--------|----------|------|
| Logistic Regression   | 0.90     | 0.87      | 0.93   | 0.90     | 0.96 |
| Decision Tree         | 0.85     | 0.85      | 0.82   | 0.84     | 0.88 |
| **Random Forest**     | **0.93** | **0.90**  | **0.96** | **0.93** | **0.96** |
| SVM                   | 0.92     | 0.87      | 0.96   | 0.92     | 0.93 |

---

## 🏆 Final Model – Random Forest Classifier
- **Chosen Model**: Random Forest Classifier  
- **Parameters**:  
  - `n_estimators = 100`  
  - `max_depth = 8`  
  - `min_samples_split = 2`  
  - `min_samples_leaf = 2`  
  - `max_features = "log2"`  
  - `bootstrap = False`  
  - `random_state = 42`  

**Performance (Test Set):**
- Accuracy: **0.93**  
- Precision: **0.90**  
- Recall: **0.96**  
- F1-score: **0.93**  
- AUC: **0.96**

**Confusion Matrix:**
```
                 Predicted Negative   Predicted Positive
Actual Negative        30                 3
Actual Positive        1                 27
```

---

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run training and evaluation**
   - Open Jupyter Notebook and click run all.

---

## 📜 License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.  
