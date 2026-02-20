import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from utils.preprocessing import clean_dataset, separate_features_label



# ======================================
# 1. Load Dataset
# ======================================

data_path = os.path.join("dataset", "phishing_dataset.csv")
df = pd.read_csv(data_path)

print("\nOriginal Dataset Shape:", df.shape)

# Clean dataset
df = clean_dataset(df)

print("After Cleaning Shape:", df.shape)

# Separate features & label
X, y = separate_features_label(df)

print("Feature Shape:", X.shape)
print("Label Distribution:\n", y.value_counts())

# ======================================
# 2. Train-Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================
# 3. Base Models
# ======================================

rf = RandomForestClassifier(random_state=42)

xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

# ======================================
# 4. Stacking Ensemble
# ======================================

stack_model = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb)
    ],
    final_estimator=LogisticRegression(),
    passthrough=True
)

# ======================================
# 5. Pipeline
# ======================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", stack_model)
])

# ======================================
# 6. Hyperparameter Tuning
# ======================================

param_grid = {
    "model__rf__n_estimators": [200],
    "model__rf__max_depth": [None, 20],
    "model__xgb__n_estimators": [200],
    "model__xgb__max_depth": [4, 6]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2
)

print("\nTraining Model...\n")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters Found:")
print(grid_search.best_params_)

# ======================================
# 7. Evaluation
# ======================================

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

roc_score = roc_auc_score(y_test, y_proba)
print("\nROC-AUC Score:", roc_score)

# ======================================
# 8. Cross Validation
# ======================================

cv_score = cross_val_score(best_model, X, y, cv=5, scoring="roc_auc")

print("\nCross Validation ROC-AUC Mean:", np.mean(cv_score))

# ======================================
# 9. Save Model
# ======================================

os.makedirs("models", exist_ok=True)

model_path = os.path.join("models", "phishguard_model.pkl")

joblib.dump(best_model, model_path)

print("\nModel Saved Successfully at:", model_path)
