# 🩺 Heart Disease Prediction Project

This project applies **Machine Learning models** to predict the presence of heart disease based on patient health data.  
The dataset is sourced from the **UCI Machine Learning Repository (Heart Disease Dataset)**.

---

## 📊 Dataset
- **Source**: UCI Heart Disease Dataset  
- **Features used**: 13 original features + one-hot encoded categorical features  
- **Final feature count**: 18  

Nominal features encoded:
```
['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
```

---

## ⚙️ Models Evaluated
1. Logistic Regression  
2. Decision Tree  
3. Random Forest  
4. Support Vector Machine (SVM)  

---

## 🏆 Best Model: Random Forest Classifier
**Parameters:**
- `n_estimators = 100`
- `max_depth = 8`
- `min_samples_split = 2`
- `min_samples_leaf = 2`
- `max_features = "log2"`
- `bootstrap = False`
- `random_state = 42`

---

## 📈 Evaluation Metrics (Test Set)

| Model                | Accuracy | Precision | Recall | F1-Score | AUC  |
|-----------------------|----------|-----------|--------|----------|------|
| Logistic Regression   | 0.90     | 0.87      | 0.93   | 0.90     | 0.96 |
| Decision Tree         | 0.85     | 0.85      | 0.82   | 0.84     | 0.88 |
| **Random Forest**     | **0.93** | **0.90**  | **0.96** | **0.93** | **0.96** |
| SVM                   | 0.92     | 0.87      | 0.96   | 0.92     | 0.93 |

---

## 🔍 Confusion Matrix (Random Forest)
```
                 Predicted Negative   Predicted Positive
Actual Negative        30                 3
Actual Positive        1                 27
```

---

## 📝 Notes
- Random Forest achieved the **best performance** across Accuracy, Recall, and F1-score.  
- Logistic Regression and SVM also performed well, with SVM excelling in recall.  
- Decision Tree was the weakest performer but remains interpretable.  
- **Preprocessing**: Nominal features encoded with `OneHotEncoder`.  
- **Scaling**: Applied only for Logistic Regression and SVM (tree-based models don’t need scaling).  
- A **pipeline (preprocessing + model)** was saved for reproducibility.  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or script to train and evaluate models.

---

## 📂 Repository Structure
```
heart-disease-prediction/
│── data/                 # Dataset (if included or linked)
│── models/               # Saved pipelines/models
│── notebooks/            # Jupyter notebooks
│── evaluation_metrics.txt # Model performance summary
│── README.md             # Project documentation
│── requirements.txt      # Dependencies
```

---

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
