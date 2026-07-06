from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TARGET_COLUMN = "PCOS (Y/N)"
DROP_COLUMNS = ["Sl. No", "Patient File No.", "Unnamed: 44"]


def resolve_paths() -> tuple[Path, Path]:
    # Define os caminhos a partir da raiz do repositório para que o script funcione a partir de qualquer diretório atual.
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = repo_root / "FASE_1" / "data" / "PCOS_data_without_infertility.xlsx"
    models_dir = repo_root / "FASE_2" / "models"
    return dataset_path, models_dir


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    # Espelha o mesmo conjunto de dados e planilha usados ​​no notebook de modelagem FASE_1.
    df = pd.read_excel(dataset_path, sheet_name="Full_new")
    cols_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Mantém o comportamento do notebook: converter valores incorretos (por exemplo, "1.99.") para NaN.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # A imputação "mediana" segue a abordagem de pré-processamento do notebook.
    null_columns = df.columns[df.isnull().any()].tolist()
    if null_columns:
        imputer = SimpleImputer(strategy="median")
        df[null_columns] = imputer.fit_transform(df[null_columns])

    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, _X_val, y_train, _y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=round(0.2 / 0.80, 4),
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return X_train, X_test, y_train, y_test


def find_best_k(X_train: pd.DataFrame, y_train: pd.Series, feature_count: int) -> int:
    # O KNN usa recursos escalonados; mantemos a mesma pontuação de validação cruzada usada no notebook (F1).
    X_train_scaled = StandardScaler().fit_transform(X_train)
    k_range = range(1, feature_count)
    errors = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring="f1")
        errors.append(1 - scores.mean())
    return list(k_range)[int(np.argmin(errors))]


def find_best_depth(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> int:
    # A profundidade da árvore de decisão é ajustada com base na F1 de validação para a classe 1 (SOP).
    depths = range(2, 12)
    val_f1 = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        val_f1.append(f1_score(y_val, preds, pos_label=1))
    return list(depths)[int(np.argmax(val_f1))]


def build_and_save_models(df: pd.DataFrame, models_dir: Path) -> dict[str, float | int]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=round(0.2 / 0.80, 4),
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    best_k = find_best_k(X_train, y_train, feature_count=df.columns.size - 1)
    best_depth = find_best_depth(X_train, y_train, X_val, y_val)

    # Treino de modelos de regressão logística (LR) e KNN de linha de base como pipelines com o StandardScaler.
    lr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    knn_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=best_k)),
        ]
    )
    dt_model = DecisionTreeClassifier(max_depth=best_depth, random_state=RANDOM_STATE)

    lr_pipeline.fit(X_train, y_train)
    knn_pipeline.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)

    # Persistência de artefatos de linha de base para reutilização na otimização de AG (Etapa 3).
    joblib.dump(lr_pipeline, models_dir / "baseline_lr.joblib")
    joblib.dump(knn_pipeline, models_dir / "baseline_knn.joblib")
    joblib.dump(dt_model, models_dir / "baseline_dt.joblib")

    metrics = {
        "lr_test_f1": f1_score(y_test, lr_pipeline.predict(X_test), pos_label=1),
        "knn_test_f1": f1_score(y_test, knn_pipeline.predict(X_test), pos_label=1),
        "dt_test_f1": f1_score(y_test, dt_model.predict(X_test), pos_label=1),
    }

    # Metadados em JSON compactos para tornar as execuções auditáveis ​​e reproduzíveis.
    metadata = {
        "random_state": RANDOM_STATE,
        "dataset_path": str(models_dir.parents[1] / "FASE_1" / "data" / "PCOS_data_without_infertility.xlsx"),
        "sheet_name": "Full_new",
        "dropped_columns": DROP_COLUMNS,
        "best_k": best_k,
        "best_depth": best_depth,
        "test_metrics": metrics,
    }
    (models_dir / "baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"best_k": best_k, "best_depth": best_depth, **metrics}


def main() -> None:
    dataset_path, models_dir = resolve_paths()
    models_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(dataset_path)
    results = build_and_save_models(df, models_dir)
    print("Baselines exportados com sucesso.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
