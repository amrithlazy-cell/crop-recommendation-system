# Crop Recommendation System — Submission README

**Subject:** Machine Learning Lab  
**Algorithm:** Random Forest Classifier (vs Logistic Regression & SVM)  
**Course Outcome:** CO5 – Solve real life problems using appropriate ML models and evaluate performance measures  

---

## File Manifest

| File | Description |
|------|-------------|
| `train_evaluate.py` | **Main ML pipeline** – loads data, preprocesses, trains LR / SVM / RF, runs 5-fold CV, evaluates all metrics, saves 5 plots and the model |
| `streamlit_app.py` | **Interactive web app** – sliders for all 7 inputs, Predict button, top-3 probabilities, preset soil scenarios |
| `generate_report.py` | Script that generates the Word report (with embedded plots) |
| `Crop_Recommendation_Report.docx` | **Lab report** (A4, 10 sections, all figures embedded) |
| `Crop_recommendation.csv` | Dataset – 2,200 samples × 8 columns (22 crop classes) |
| `crop_recommendation_model.joblib` | Serialised Random Forest pipeline + LabelEncoder |
| `requirements.txt` | Python package versions |
| `plots/` | 5 saved PNG figures (300 dpi) |

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the ML pipeline
```bash
python train_evaluate.py
```
Outputs: metrics table in terminal, 5 plots in `plots/`, model saved as `.joblib`

### 3. Launch the Streamlit app
```bash
streamlit run streamlit_app.py
```
> Make sure `crop_recommendation_model.joblib` exists (run step 2 first)

### 4. (Re-)generate the Word report
```bash
python generate_report.py
```

---

## Key Results

| Algorithm | 5-Fold CV Accuracy | Test Accuracy | F1 (Weighted) |
|---|---|---|---|
| Logistic Regression | 96.82 % | 97.27 % | 97.25 % |
| SVM (RBF) | 97.90 % | 98.41 % | 98.40 % |
| **Random Forest ✓** | **99.26 %** | **99.55 %** | **99.55 %** |

**Winner:** Random Forest (selected by 5-fold CV, not test score)

---

## Plots Generated
1. `class_distribution.png` – bar chart of 22 balanced crop classes  
2. `correlation_heatmap.png` – Pearson correlation of 7 features  
3. `confusion_matrix.png` – 22×22 confusion matrix for Random Forest  
4. `roc_curve.png` – Multi-class OvR ROC curves (micro-AUC = 1.000)  
5. `model_comparison.png` – side-by-side metric bar chart (all 3 models)

