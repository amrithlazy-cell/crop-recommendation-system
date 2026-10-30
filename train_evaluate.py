"""
Crop Recommendation System using Machine Learning
Comparing Logistic Regression, Support Vector Machine (SVM), and Random Forest Classifier.
Author: Antigravity Assistant
Course Outcome: CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# -------------------------------------------------------------
# Configuration and Setup
# -------------------------------------------------------------
DATA_PATH = "Crop_recommendation.csv"
PLOTS_DIR = "plots"
MODEL_SAVE_PATH = "crop_recommendation_model.joblib"
RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

print("=" * 80)
print("             CROP RECOMMENDATION SYSTEM - ML PIPELINE")
print("=" * 80)

# -------------------------------------------------------------
# Step 3.1: Load data and print shape, columns, class counts
# -------------------------------------------------------------
print("\n[Step 1/6] Loading Dataset...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset '{DATA_PATH}' not found. Please ensure it is in the root directory.")

df = pd.read_csv(DATA_PATH)
print(f"-> Dataset loaded successfully.")
print(f"-> Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"-> Column Names: {list(df.columns)}")
print("\n-> Class Value Counts:")
print(df["label"].value_counts())

# -------------------------------------------------------------
# Step 3.2: Preprocessing
# -------------------------------------------------------------
print("\n[Step 2/6] Preprocessing and Data Integrity Checks...")
missing_vals = df.isnull().sum()
duplicates_count = df.duplicated().sum()

print("-> Missing values per column:")
print(missing_vals)
print(f"-> Initial duplicate rows detected: {duplicates_count}")

# Explicitly remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)
print(f"-> Duplicates removed. Clean dataset shape: {df.shape}")

# Encode text target labels
label_encoder = LabelEncoder()
df["encoded_label"] = label_encoder.fit_transform(df["label"])
class_names = list(label_encoder.classes_)
n_classes = len(class_names)
print(f"-> Encoded {n_classes} distinct crop classes.")

# Separate feature matrix X and target y
feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[feature_cols].copy()
y = df["encoded_label"].copy()

# Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"-> Stratified Train-Test Split (80/20):")
print(f"   Train samples: {X_train.shape[0]} (Features: {X_train.shape[1]})")
print(f"   Test samples:  {X_test.shape[0]} (Features: {X_test.shape[1]})")

# -------------------------------------------------------------
# Step 3.3: Algorithm Selection and Pipeline Construction
# -------------------------------------------------------------
print("\n[Step 3/6] Algorithm Selection and Pipeline Definition...")
"""
Algorithm Selection Rationale:
1. Logistic Regression:
   - Role: Linear baseline model.
   - Justification: Provides high interpretability, calibrated class probabilities, and serves
     as a mathematical benchmark to determine if linear boundaries suffice for crop classification.
2. Support Vector Machine (SVC with RBF Kernel):
   - Role: Non-linear margin-maximization classifier.
   - Justification: Projects continuous soil/weather features into high-dimensional space
     using the Radial Basis Function kernel to separate non-linear clusters while maximizing geometric margins.
3. Random Forest Classifier:
   - Role: Non-linear ensemble tree-based bagging model.
   - Justification: Tabular agricultural variables feature complex non-linear interactions
     (e.g., high N combined with high rainfall vs low humidity). Random Forest captures high-order
     interactions without overfitting, resists outliers, and provides robust multi-class decisions.
"""

algorithms = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "Support Vector Machine": CalibratedClassifierCV(
        SVC(kernel="rbf", random_state=RANDOM_STATE), ensemble=False
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=RANDOM_STATE
    ),
}

# Wrap models in scikit-learn Pipelines with StandardScaler to prevent data leakage
pipelines = {}
for name, model in algorithms.items():
    pipelines[name] = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

# -------------------------------------------------------------
# Step 3.4: 5-Fold Cross-Validation on Training Set
# -------------------------------------------------------------
print("\n[Step 4/6] Performing 5-Fold Stratified Cross-Validation on Training Data...")
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

for name, pipe in pipelines.items():
    print(f"   Evaluating {name} with 5-fold CV...", end=" ", flush=True)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv_strategy, scoring="accuracy")
    cv_results[name] = {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std()
    }
    print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Selection of the winning model based solely on 5-fold CV score
best_model_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
print(f"\n=> Final Selected Model (based on 5-fold CV, not test score): **{best_model_name}**")
print(f"   CV Score: {cv_results[best_model_name]['mean']:.4f} (+/- {cv_results[best_model_name]['std']:.4f})")

# -------------------------------------------------------------
# Step 3.5: Training on Full Train Set & Testing on Test Set
# -------------------------------------------------------------
print("\n[Step 5/6] Training on Training Set and Evaluating on 20% Held-Out Test Set...")

# Binarize test targets for multi-class ROC-AUC
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

evaluation_metrics = {}
predictions = {}
predicted_probabilities = {}

for name, pipe in pipelines.items():
    print(f"   Fitting {name}...", end=" ", flush=True)
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    
    predictions[name] = y_pred
    predicted_probabilities[name] = y_proba
    
    acc = accuracy_score(y_test, y_pred)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="weighted")
    except Exception as e:
        roc_auc = np.nan
        print(f"(ROC-AUC calculation note: {e})", end=" ")
        
    evaluation_metrics[name] = {
        "5-Fold CV Accuracy": cv_results[name]["mean"],
        "CV Std Dev": cv_results[name]["std"],
        "Test Accuracy": acc,
        "Precision (Weighted)": prec_weighted,
        "Recall (Weighted)": rec_weighted,
        "F1-Score (Weighted)": f1_weighted,
        "Precision (Macro)": prec_macro,
        "Recall (Macro)": rec_macro,
        "F1-Score (Macro)": f1_macro,
        "ROC-AUC (OvR Weighted)": roc_auc
    }
    print("Done.")

# Display Metrics Table
metrics_df = pd.DataFrame(evaluation_metrics).T
print("\n" + "=" * 90)
print("                           MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 90)
print(metrics_df[[
    "5-Fold CV Accuracy", "Test Accuracy", "Precision (Weighted)", 
    "Recall (Weighted)", "F1-Score (Weighted)", "ROC-AUC (OvR Weighted)"
]].to_string())
print("=" * 90)

print(f"\nDetailed Classification Report for Winner ({best_model_name}):")
print(classification_report(y_test, predictions[best_model_name], target_names=class_names, digits=4))

# Save the winning model bundle for Streamlit app
print(f"-> Saving winning model bundle ({best_model_name}) to '{MODEL_SAVE_PATH}'...")
model_bundle = {
    "model_name": best_model_name,
    "pipeline": pipelines[best_model_name],
    "label_encoder": label_encoder,
    "feature_names": feature_cols,
    "class_names": class_names,
    "cv_accuracy": cv_results[best_model_name]["mean"],
    "test_accuracy": evaluation_metrics[best_model_name]["Test Accuracy"]
}
joblib.dump(model_bundle, MODEL_SAVE_PATH)
print("-> Model artifact saved successfully.")

# -------------------------------------------------------------
# Step 3.6: Generate and Save Required Visualizations
# -------------------------------------------------------------
print("\n[Step 6/6] Generating and Saving Visualizations in 'plots/'...")

# Graph 1: Class Distribution
plt.figure(figsize=(12, 6))
counts = df["label"].value_counts()
sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
plt.title("Figure 1: Target Class Distribution Across Crop Categories", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Crop Label", fontsize=12, labelpad=8)
plt.ylabel("Number of Samples", fontsize=12, labelpad=8)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
fig1_path = os.path.join(PLOTS_DIR, "class_distribution.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"   [1/5] Saved class distribution plot to {fig1_path}")

# Graph 2: Feature Correlation Heatmap
plt.figure(figsize=(9, 7))
corr_matrix = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="Spectral", 
    mask=mask,
    cbar_kws={"label": "Pearson Correlation Coefficient"}, 
    linewidths=0.5
)
plt.title("Figure 2: Correlation Heatmap of Soil Nutrients and Climatic Factors", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
fig2_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"   [2/5] Saved correlation heatmap to {fig2_path}")

# Graph 3: Confusion Matrix for Winning Model
plt.figure(figsize=(14, 11))
cm = confusion_matrix(y_test, predictions[best_model_name])
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=class_names, 
    yticklabels=class_names,
    cbar=True,
    linewidths=0.2
)
plt.title(f"Figure 3: Confusion Matrix - Best Model ({best_model_name}) on Held-Out Test Set", fontsize=14, fontweight="bold", pad=14)
plt.xlabel("Predicted Crop Label", fontsize=12, labelpad=10)
plt.ylabel("Ground Truth Crop Label", fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
fig3_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"   [3/5] Saved confusion matrix to {fig3_path}")

# Graph 4: Multi-Class ROC Curves (OvR) for Best Model
plt.figure(figsize=(10, 8))
y_score = predicted_probabilities[best_model_name]

# Compute micro-average ROC curve
fpr_dict = dict()
tpr_dict = dict()
roc_auc_dict = dict()

for i in range(n_classes):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

# Compute micro-average ROC curve and ROC area
fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

# Plot micro-average
plt.plot(
    fpr_dict["micro"], tpr_dict["micro"],
    label=f"Micro-average ROC (AUC = {roc_auc_dict['micro']:.4f})",
    color="deeppink", linestyle=":", linewidth=3
)

# Plot selected representative classes to keep the plot readable and informative
sample_indices = [0, 4, 8, 12, 16, 20]
colors = ["navy", "darkorange", "forestgreen", "crimson", "purple", "teal"]
for idx, col in zip(sample_indices, colors):
    plt.plot(
        fpr_dict[idx], tpr_dict[idx],
        color=col, lw=1.8,
        label=f"{class_names[idx]} (AUC = {roc_auc_dict[idx]:.3f})"
    )

plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.50)")
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=8)
plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, labelpad=8)
plt.title(f"Figure 4: Multi-Class ROC-AUC Curves - {best_model_name}", fontsize=14, fontweight="bold", pad=12)
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
fig4_path = os.path.join(PLOTS_DIR, "roc_curve.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"   [4/5] Saved ROC curve to {fig4_path}")

# Graph 5: Model Comparison Bar Chart
plt.figure(figsize=(11, 6))
metrics_to_plot = ["5-Fold CV Accuracy", "Test Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)"]
plot_data = metrics_df[metrics_to_plot].reset_index().melt(
    id_vars="index", var_name="Metric", value_name="Score"
)
plot_data = plot_data.rename(columns={"index": "Model"})

sns.barplot(data=plot_data, x="Metric", y="Score", hue="Model", palette="Set2")
plt.title("Figure 5: Quantitative Model Comparison Across Evaluation Metrics", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Evaluation Metric", fontsize=12, labelpad=8)
plt.ylabel("Score (0.0 to 1.0)", fontsize=12, labelpad=8)
plt.ylim([0.85, 1.02])
plt.xticks(rotation=15, ha="right", fontsize=10)
plt.legend(title="Algorithm", loc="lower left", fontsize=10)
plt.tight_layout()
fig5_path = os.path.join(PLOTS_DIR, "model_comparison.png")
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"   [5/5] Saved model comparison bar chart to {fig5_path}")

print("\n" + "=" * 80)
print("Pipeline execution complete! All models trained, evaluated, and artifacts saved.")
print("=" * 80)

