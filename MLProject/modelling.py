# IMPORT LIBRARIES

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# LOAD DATASET

df = pd.read_csv("adult_preprocessed.csv")

print("Dataset berhasil dimuat.")
print(df.head())
print(f"\nUkuran dataset : {df.shape}")

X = df.drop("income", axis=1)
y = df["income"]

# SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData Train :", X_train.shape)
print("Data Test  :", X_test.shape)

# MLFLOW AUTOLOG

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Adult Income Classification")

mlflow.sklearn.autolog()

# TRAINING MODEL

with mlflow.start_run():

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n===== HASIL EVALUASI =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    

