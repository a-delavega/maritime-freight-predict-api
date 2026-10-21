"""
Clean ML Training Pipeline for Maritime Exercise Difficulty Classification.

- No SMOTE (avoids synthetic artifacts).
- Strict train/test split before feature scaling (avoids Data Leakage).
- Exports model to JSON format (native XGBoost) and encoders/metadata to model_assets/.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "labeled"
ASSETS_DIR = PROJECT_ROOT / "model_assets"

# Canonical ordinal mapping: Physical severity monotonicity
SITUATION_MAPPING = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}

EXERCISE_CLASS_MAPPING = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}

INVERSE_EXERCISE_MAPPING = {v: k for k, v in EXERCISE_CLASS_MAPPING.items()}


def load_labeled_data(folder: Path) -> pd.DataFrame:
    """Load and combine all labeled exercise CSV files."""
    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} situations across {len(csv_files)} exercise CSV files.")
    return df


def preprocess_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Map categorical difficulties to semantic ordinal numbers and aggregate by Exercise."""
    spanish_to_english = {
        "BAJA": "LOW",
        "MEDIA": "MEDIUM",
        "ALTA": "HIGH",
    }
    df["SituationDifficulty"] = df["DificultadSituacion"].map(spanish_to_english)
    df["ExerciseDifficulty"] = df["DificultadEjercicio"].map(spanish_to_english)

    df["SituationDifficultyEncoded"] = df["SituationDifficulty"].map(SITUATION_MAPPING)
    df["ExerciseDifficultyEncoded"] = df["ExerciseDifficulty"].map(EXERCISE_CLASS_MAPPING)

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
    return data


def train(data: pd.DataFrame) -> tuple[XGBClassifier, StandardScaler]:
    """Train XGBoost model without SMOTE and without data leakage."""
    features = ["mean_difficulty", "max_difficulty", "min_difficulty", "situation_count"]
    X = data[features]
    y = data["ExerciseDifficultyEncoded"]

    # 1. Stratified Train / Test Split FIRST (No Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Fit scaler ONLY on X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model setup with class weighting handled natively by XGBoost if necessary
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
    )

    # Cross-validation on train split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1_macro")
    print(f"5-Fold CV Macro F1 on Train: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Fit on full train split
    model.fit(X_train_scaled, y_train)

    # 4. Final Evaluation on Holdout Test Set
    y_pred = model.predict(X_test_scaled)
    target_names = [INVERSE_EXERCISE_MAPPING[i] for i in range(3)]

    print("\n--- Holdout Test Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, scaler


def save_assets(model: XGBClassifier, scaler: StandardScaler) -> None:
    """Save XGBoost native model (.json) and scaler/metadata to model_assets/."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Save native XGBoost Booster directly to JSON (bypasses sklearn wrapper bug, zero pickle)
    model_json_path = ASSETS_DIR / "model.json"
    booster = model.get_booster()
    booster.save_model(str(model_json_path))
    print(f"Saved native XGBoost Booster to: {model_json_path}")

    # Save scaler parameters as pure JSON to eliminate joblib/pickle completely
    scaler_metadata = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "features": ["mean_difficulty", "max_difficulty", "min_difficulty", "situation_count"],
    }
    scaler_path = ASSETS_DIR / "scaler_params.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_metadata, f, indent=2)
    print(f"Saved scaler parameters to: {scaler_path}")

    # Save domain encoding metadata
    encoders_metadata = {
        "situation_mapping": SITUATION_MAPPING,
        "exercise_class_mapping": EXERCISE_CLASS_MAPPING,
        "exercise_decoding": INVERSE_EXERCISE_MAPPING,
    }
    metadata_path = ASSETS_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(encoders_metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")


def main():
    print("=" * 60)
    print("Clean ML Pipeline: Maritime Navigation Exercise Classifier")
    print("=" * 60)

    if not any(DATA_DIR.glob("*.csv")):
        print(f"\n[Notice] No CSV files found in {DATA_DIR} yet.")
        print("Please place your labeled CSV files inside data/labeled/ and rerun this script.")
        return

    data = load_labeled_data(DATA_DIR)
    aggregated_data = preprocess_and_aggregate(data)
    model, scaler = train(aggregated_data)
    save_assets(model, scaler)
    print("\nTraining and export to model_assets/ completed successfully!")


if __name__ == "__main__":
    main()
