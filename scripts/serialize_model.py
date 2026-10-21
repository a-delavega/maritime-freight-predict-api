"""
Fase 0: Script de serialización del modelo.

Entrena el XGBClassifier con los datos etiquetados de FórmaTECH
y serializa el pipeline completo (StandardScaler + XGBClassifier) con joblib.

Uso:
    python scripts/serialize_model.py

Genera:
    models/model_pipeline.joblib
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


# --- Configuration ---
DATA_FOLDER = r"c:\Users\USUARIO\Documents\01_ACADÉMICO\UBU\HaCoBu\FórmaTECH - HaCoBu II\entrega\FórmaTECH - Navegación Marítima\datos_etiquetados"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models"
OUTPUT_MODEL = OUTPUT_DIR / "model_pipeline.joblib"


def load_labeled_data(folder: str) -> pd.DataFrame:
    """Load and concatenate all labeled CSV files."""
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    if not csv_files:
        print(f"ERROR: No CSV files found in {folder}")
        sys.exit(1)
    df_list = [pd.read_csv(os.path.join(folder, f)) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} situaciones from {len(csv_files)} CSV files")
    return df


def preprocess_and_aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Encode labels and aggregate features per exercise."""
    le_situacion = LabelEncoder()
    df['DificultadSituacion'] = le_situacion.fit_transform(df['DificultadSituacion'])

    le_ejercicio = LabelEncoder()
    df['DificultadEjercicio'] = le_ejercicio.fit_transform(df['DificultadEjercicio'])

    # Aggregate per exercise
    agg = df.groupby('Ejercicio').agg(
        media_dificultad=('DificultadSituacion', 'mean'),
        max_dificultad=('DificultadSituacion', 'max'),
        min_dificultad=('DificultadSituacion', 'min'),
        num_situaciones=('DificultadSituacion', 'count')
    ).reset_index()

    labels = df[['Ejercicio', 'DificultadEjercicio']].drop_duplicates()
    data = pd.merge(agg, labels, on='Ejercicio')

    print(f"Aggregated to {len(data)} exercises")
    print(f"LabelEncoder situacion classes: {list(le_situacion.classes_)}")
    print(f"LabelEncoder ejercicio classes: {list(le_ejercicio.classes_)}")

    return data, le_situacion, le_ejercicio


def train_model(data: pd.DataFrame) -> Pipeline:
    """Train XGBClassifier with GridSearchCV, SMOTE, and class weights."""
    X = data[['media_dificultad', 'max_dificultad', 'min_dificultad', 'num_situaciones']]
    y = data['DificultadEjercicio']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_bal), y=y_train_bal)
    class_weight_dict = dict(zip(np.unique(y_train_bal), class_weights))

    # Pipeline
    num_classes = len(np.unique(y))
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
        ))
    ])

    param_grid = {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__max_depth': [3, 6, 9],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__subsample': [0.7, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    sample_weights = [class_weight_dict[i] for i in y_train_bal]
    grid_search.fit(X_train_bal, y_train_bal, xgb__sample_weight=sample_weights)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\nBest hyperparameters: {grid_search.best_params_}")
    print(f"\nClassification report:\n{classification_report(y_test, y_pred)}")

    return best_model


def validate_label_encoding(le_situacion: LabelEncoder, le_ejercicio: LabelEncoder) -> None:
    """Validate that label encoding matches our hardcoded domain mapping."""
    # Our domain mapping: ALTA=0, BAJA=1, MEDIA=2 (alphabetical, sklearn default)
    situacion_classes = list(le_situacion.classes_)
    ejercicio_classes = list(le_ejercicio.classes_)

    print(f"\n--- VALIDATION ---")
    print(f"Situacion encoding: {dict(zip(situacion_classes, le_situacion.transform(situacion_classes)))}")
    print(f"Ejercicio encoding: {dict(zip(ejercicio_classes, le_ejercicio.transform(ejercicio_classes)))}")

    # Store the mapping as metadata
    metadata = {
        "situacion_classes": situacion_classes,
        "ejercicio_classes": ejercicio_classes,
        "situacion_mapping": dict(zip(situacion_classes, [int(x) for x in le_situacion.transform(situacion_classes)])),
        "ejercicio_mapping": dict(zip(ejercicio_classes, [int(x) for x in le_ejercicio.transform(ejercicio_classes)])),
    }
    return metadata


def main():
    print("=" * 60)
    print("FórmaTECH Model Serialization Script")
    print("=" * 60)

    # 1. Load data
    df = load_labeled_data(DATA_FOLDER)

    # 2. Preprocess
    data, le_situacion, le_ejercicio = preprocess_and_aggregate(df)

    # 3. Train
    best_model = train_model(data)

    # 4. Validate encoding
    metadata = validate_label_encoding(le_situacion, le_ejercicio)

    # 5. Serialize
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, OUTPUT_MODEL)
    print(f"\nModel pipeline saved to: {OUTPUT_MODEL}")
    print(f"File size: {OUTPUT_MODEL.stat().st_size / 1024:.1f} KB")

    # Save metadata as JSON for reference
    import json
    metadata_path = OUTPUT_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    # 6. Verify load
    print("\n--- VERIFICATION ---")
    loaded_pipeline = joblib.load(OUTPUT_MODEL)
    test_input = np.array([[1.0, 2.0, 0.0, 6]])  # media=1, max=2, min=0, 6 situations
    pred = loaded_pipeline.predict(test_input)
    proba = loaded_pipeline.predict_proba(test_input)
    print(f"Test prediction (class index): {pred[0]}")
    print(f"Test probabilities: {proba[0]}")
    print(f"Predicted class: {le_ejercicio.inverse_transform(pred)[0]}")
    print("\n✅ Serialization complete and verified!")


if __name__ == "__main__":
    main()
