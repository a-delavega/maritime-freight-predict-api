"""
Model serialization script.

Trains the XGBClassifier using labeled maritime simulation data
and serializes the complete pipeline (StandardScaler + XGBClassifier) via joblib.

Usage:
    python scripts/serialize_model.py

Outputs:
    models/model_pipeline.joblib
"""

import json
import os
from pathlib import Path
import sys
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


# --- Configuration ---
DATA_FOLDER = r"c:\Users\USUARIO\Documents\01_ACADÉMICO\UBU\HaCoBu\FórmaTECH - HaCoBu II\entrega\FórmaTECH - Navegación Marítima\datos_etiquetados"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models"
OUTPUT_MODEL = OUTPUT_DIR / "model_pipeline.joblib"


def load_labeled_data(folder: str) -> pd.DataFrame:
    """Load and concatenate all labeled CSV files."""
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not csv_files:
        print(f"ERROR: No CSV files found in {folder}")
        sys.exit(1)
    df_list = [pd.read_csv(os.path.join(folder, f)) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} situations from {len(csv_files)} CSV files")
    return df


def preprocess_and_aggregate(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Encode labels and aggregate features per exercise."""
    # Map Spanish dataset labels to English domain values
    # Original labels: BAJA, MEDIA, ALTA
    # Target English: LOW, MEDIUM, HIGH
    spanish_to_english = {
        "ALTA": "HIGH",
        "BAJA": "LOW",
        "MEDIA": "MEDIUM",
    }
    df["SituationDifficulty"] = df["DificultadSituacion"].map(spanish_to_english)
    df["ExerciseDifficulty"] = df["DificultadEjercicio"].map(spanish_to_english)

    label_encoder_situation = LabelEncoder()
    # Alphabetical order: HIGH=0, LOW=1, MEDIUM=2
    df["SituationDifficultyEncoded"] = label_encoder_situation.fit_transform(
        df["SituationDifficulty"]
    )

    label_encoder_exercise = LabelEncoder()
    df["ExerciseDifficultyEncoded"] = label_encoder_exercise.fit_transform(
        df["ExerciseDifficulty"]
    )

    # Aggregate per exercise
    aggregated = (
        df.groupby("Ejercicio")
        .agg(
            mean_difficulty=("SituationDifficultyEncoded", "mean"),
            max_difficulty=("SituationDifficultyEncoded", "max"),
            min_difficulty=("SituationDifficultyEncoded", "min"),
            situation_count=("SituationDifficultyEncoded", "count"),
        )
        .reset_index()
    )

    labels = df[["Ejercicio", "ExerciseDifficultyEncoded"]].drop_duplicates()
    data = pd.merge(aggregated, labels, on="Ejercicio")

    print(f"Aggregated to {len(data)} exercises")
    print(f"LabelEncoder situation classes: {list(label_encoder_situation.classes_)}")
    print(f"LabelEncoder exercise classes: {list(label_encoder_exercise.classes_)}")

    return data, label_encoder_situation, label_encoder_exercise


def train_model(data: pd.DataFrame) -> Pipeline:
    """Train XGBClassifier with GridSearchCV, SMOTE, and class weights."""
    features = [
        "mean_difficulty",
        "max_difficulty",
        "min_difficulty",
        "situation_count",
    ]
    X = data[features]
    y = data["ExerciseDifficultyEncoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Class weights
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train_bal), y=y_train_bal
    )
    class_weight_dict = dict(zip(np.unique(y_train_bal), class_weights))

    num_classes = len(np.unique(y))
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                XClassifier := XGBClassifier(
                    objective="multi:softprob",
                    num_class=num_classes,
                ),
            ),
        ]
    )

    param_grid = {
        "xgb__n_estimators": [50, 100, 200],
        "xgb__max_depth": [3, 6, 9],
        "xgb__learning_rate": [0.01, 0.1, 0.2],
        "xgb__subsample": [0.7, 1.0],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    sample_weights = [class_weight_dict[i] for i in y_train_bal]
    grid_search.fit(X_train_bal, y_train_bal, xgb__sample_weight=sample_weights)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\nBest hyperparameters: {grid_search.best_params_}")
    print(f"\nClassification report:\n{classification_report(y_test, y_pred)}")

    return best_model


def validate_label_encoding(
    label_encoder_situation: LabelEncoder, label_encoder_exercise: LabelEncoder
) -> dict:
    """Validate that label encoding matches our hardcoded domain mapping."""
    situation_classes = list(label_encoder_situation.classes_)
    exercise_classes = list(label_encoder_exercise.classes_)

    print("\n--- VALIDATION ---")
    print(
        f"Situation encoding: {dict(zip(situation_classes, label_encoder_situation.transform(situation_classes)))}"
    )
    print(
        f"Exercise encoding: {dict(zip(exercise_classes, label_encoder_exercise.transform(exercise_classes)))}"
    )

    metadata = {
        "situation_classes": situation_classes,
        "exercise_classes": exercise_classes,
        "situation_mapping": dict(
            zip(
                situation_classes,
                [int(x) for x in label_encoder_situation.transform(situation_classes)],
            )
        ),
        "exercise_mapping": dict(
            zip(
                exercise_classes,
                [int(x) for x in label_encoder_exercise.transform(exercise_classes)],
            )
        ),
    }
    return metadata


def main():
    print("=" * 60)
    print("Maritime Navigation Model Serialization Script")
    print("=" * 60)

    # 1. Load data
    df = load_labeled_data(DATA_FOLDER)

    # 2. Preprocess
    data, le_situation, le_exercise = preprocess_and_aggregate(df)

    # 3. Train
    best_model = train_model(data)

    # 4. Validate encoding
    metadata = validate_label_encoding(le_situation, le_exercise)

    # 5. Serialize
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, OUTPUT_MODEL)
    print(f"\nModel pipeline saved to: {OUTPUT_MODEL}")
    print(f"File size: {OUTPUT_MODEL.stat().st_size / 1024:.1f} KB")

    metadata_path = OUTPUT_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    # 6. Verify load
    print("\n--- VERIFICATION ---")
    loaded_pipeline = joblib.load(OUTPUT_MODEL)
    test_input = np.array([[1.0, 2.0, 0.0, 6]])  # mean=1, max=2, min=0, count=6
    pred = loaded_pipeline.predict(test_input)
    proba = loaded_pipeline.predict_proba(test_input)
    print(f"Test prediction (class index): {pred[0]}")
    print(f"Test probabilities: {proba[0]}")
    print(f"Predicted class: {le_exercise.inverse_transform(pred)[0]}")
    print("\nSerialization complete and verified!")


if __name__ == "__main__":
    main()
